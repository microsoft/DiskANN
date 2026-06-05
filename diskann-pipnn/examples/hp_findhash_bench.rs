//! Microbench: find_hash variants on a reservoir of 8-byte ReservoirEntries.
//!
//! Compares:
//!   A: current unsorted linear AVX-512 scan (8-way vpcmpeqq, OR-aggregate)
//!   B: sorted vec + std binary_search (scalar)
//!   C: sorted vec + SIMD bisect (16-way u16 cmp per step)
//!
//! Tests for both HIT (~95% of HP's hot lookup pattern is collision-existing)
//! and MISS (the other 5%) at l_max ∈ {32, 64}.
//!
//! Usage:
//!   cargo run --release --example hp_findhash_bench -- [reps=10_000_000]

use std::env;
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone)]
struct Entry {
    neighbor: u32,
    hash: u16,
    distance: u16,
}

// LCG random
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_u16(&mut self) -> u16 { self.next_u32() as u16 }
}

fn gen_unique_hashes(n: usize, rng: &mut Rng) -> Vec<u16> {
    let mut seen = std::collections::HashSet::<u16>::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let h = rng.next_u16();
        if seen.insert(h) { out.push(h); }
    }
    out
}

fn make_entries(hashes: &[u16]) -> Vec<Entry> {
    hashes.iter().enumerate().map(|(i, &h)| Entry {
        neighbor: i as u32,
        hash: h,
        distance: 0,
    }).collect()
}

// ----- Variant A: current unsorted linear AVX-512 scan -----
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn find_unsorted_simd(entries: &[Entry], hash: u16) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = entries.len();
    if n >= 8 {
        let ptr = entries.as_ptr() as *const u64;
        let target = _mm512_set1_epi64(((hash as u64) << 32) as i64);
        let mask = _mm512_set1_epi64(0x0000FFFF_00000000u64 as i64);
        let chunks = n / 8;
        let mut found: u64 = 0;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let data = _mm512_loadu_si512(ptr.add(base) as *const __m512i);
            let masked = _mm512_and_si512(data, mask);
            let cmp = _mm512_cmpeq_epi64_mask(masked, target);
            found |= (cmp as u64) << (chunk * 8);
        }
        // tail
        let tail = n - chunks * 8;
        if tail > 0 {
            let kmask: u8 = (1u8 << tail) - 1;
            let base = chunks * 8;
            let data = _mm512_maskz_loadu_epi64(kmask, ptr.add(base) as *const i64);
            let masked = _mm512_and_si512(data, mask);
            let cmp = _mm512_cmpeq_epi64_mask(masked, target) & kmask;
            found |= (cmp as u64) << base;
        }
        if found != 0 {
            return Some(found.trailing_zeros() as usize);
        }
        return None;
    }
    // small tail (n<8): scalar
    for (i, e) in entries.iter().enumerate() {
        if e.hash == hash { return Some(i); }
    }
    None
}

// ----- Variant B: sorted vec + std binary_search (scalar) -----
#[inline(never)]
fn find_sorted_binary(entries: &[Entry], hash: u16) -> Option<usize> {
    entries.binary_search_by_key(&hash, |e| e.hash).ok()
}

// ----- Variant C: sorted vec + SIMD bisect -----
// Load 16 u16 hashes at a time, find the bucket containing `hash` via
// vpcmpeqw + first-true-lane, then refine.
#[inline(never)]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn find_sorted_simd_16wide(entries: &[Entry], hash: u16) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = entries.len();
    if n == 0 { return None; }
    // Gather hashes into contiguous u16 array first (stride 8 bytes per Entry,
    // hash is at offset 4-5). For sorted layout we keep them packed; this
    // mirrors the AoSoA hot/cold split where hashes live in their own array.
    // For benchmark fairness we pre-pack outside the timed loop.
    let hashes_ptr = entries.as_ptr() as *const u64;
    let target = _mm512_set1_epi64(((hash as u64) << 32) as i64);
    let mask = _mm512_set1_epi64(0x0000FFFF_00000000u64 as i64);
    let chunks = n / 8;
    let mut found: u64 = 0;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let data = _mm512_loadu_si512(hashes_ptr.add(base) as *const __m512i);
        let masked = _mm512_and_si512(data, mask);
        let cmp = _mm512_cmpeq_epi64_mask(masked, target);
        found |= (cmp as u64) << (chunk * 8);
    }
    let tail = n - chunks * 8;
    if tail > 0 {
        let kmask: u8 = (1u8 << tail) - 1;
        let base = chunks * 8;
        let data = _mm512_maskz_loadu_epi64(kmask, hashes_ptr.add(base) as *const i64);
        let masked = _mm512_and_si512(data, mask);
        let cmp = _mm512_cmpeq_epi64_mask(masked, target) & kmask;
        found |= (cmp as u64) << base;
    }
    if found != 0 { Some(found.trailing_zeros() as usize) } else { None }
}

// ----- Variant D: sorted u16-packed hash array + 32-way SIMD scan -----
// AoSoA-style: hashes live in their own [u16] array. Scan 32-wide via
// _mm512_cmpeq_epi16_mask, much denser than the 8-way per-Entry pattern.
#[inline(never)]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn find_packed_simd_32wide(hashes: &[u16], target_hash: u16) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = hashes.len();
    let target = _mm512_set1_epi16(target_hash as i16);
    let chunks = n / 32;
    let mut found: u64 = 0;
    for chunk in 0..chunks {
        let base = chunk * 32;
        let data = _mm512_loadu_si512(hashes.as_ptr().add(base) as *const __m512i);
        let cmp = _mm512_cmpeq_epi16_mask(data, target);
        found |= (cmp as u64) << (chunk * 32);
    }
    let tail = n - chunks * 32;
    if tail > 0 {
        let kmask: u32 = (1u32 << tail) - 1;
        let base = chunks * 32;
        let data = _mm512_maskz_loadu_epi16(kmask, hashes.as_ptr().add(base) as *const i16);
        let cmp = _mm512_cmpeq_epi16_mask(data, target) & kmask;
        found |= (cmp as u64) << base;
    }
    if found != 0 { Some(found.trailing_zeros() as usize) } else { None }
}

// ----- Variant E: sorted packed u16 hashes + binary search -----
#[inline(never)]
fn find_packed_binary(hashes: &[u16], target_hash: u16) -> Option<usize> {
    hashes.binary_search(&target_hash).ok()
}

fn run_bench(l_max: usize, reps: usize) {
    let mut rng = Rng::new(0xC0DEC0DE_F00DBEEF);
    let hashes_random = gen_unique_hashes(l_max, &mut rng);
    let entries_unsorted = make_entries(&hashes_random);

    let mut hashes_sorted = hashes_random.clone();
    hashes_sorted.sort_unstable();
    let entries_sorted = make_entries(&hashes_sorted);

    // Query workload: 50% HIT (uniformly from the array), 50% MISS (random other).
    let n_queries = reps;
    let mut queries: Vec<u16> = Vec::with_capacity(n_queries);
    let mut expect_hit: Vec<bool> = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        if i & 1 == 0 {
            // HIT: pick random element from sorted array
            let idx = (rng.next_u32() as usize) % l_max;
            queries.push(hashes_sorted[idx]);
            expect_hit.push(true);
        } else {
            // MISS: random hash; if it accidentally matches just take it (rare)
            queries.push(rng.next_u16());
            expect_hit.push(false);
        }
    }

    let bench = |name: &str, mut f: Box<dyn FnMut(usize) -> Option<usize>>| {
        let mut acc: usize = 0;
        let t = Instant::now();
        for i in 0..n_queries {
            let r = f(i);
            acc = acc.wrapping_add(r.unwrap_or(0));
        }
        let dt = t.elapsed();
        let ns_per = dt.as_nanos() as f64 / n_queries as f64;
        println!("  {:28}: {:6.2}ns/call ({:8} reps, sink={})", name, ns_per, n_queries, acc & 0xFF);
    };

    println!("\n=== l_max={l_max}, reps={n_queries} ===");

    let e = entries_unsorted.clone();
    let q = queries.clone();
    bench("A unsorted SIMD 8-way", Box::new(move |i| unsafe {
        find_unsorted_simd(&e, q[i])
    }));

    let e = entries_sorted.clone();
    let q = queries.clone();
    bench("B sorted binary scalar", Box::new(move |i| {
        find_sorted_binary(&e, q[i])
    }));

    let e = entries_sorted.clone();
    let q = queries.clone();
    bench("C sorted SIMD 8-way", Box::new(move |i| unsafe {
        find_sorted_simd_16wide(&e, q[i])
    }));

    let h = hashes_sorted.clone();
    let q = queries.clone();
    bench("D packed SIMD 32-way", Box::new(move |i| unsafe {
        find_packed_simd_32wide(&h, q[i])
    }));

    let h = hashes_sorted.clone();
    let q = queries.clone();
    bench("E packed binary scalar", Box::new(move |i| {
        find_packed_binary(&h, q[i])
    }));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    for l_max in [16usize, 32, 48, 64, 96, 128] {
        run_bench(l_max, reps);
    }
}
