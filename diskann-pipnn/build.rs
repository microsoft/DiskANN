fn main() {
    let search_paths = [
        "/usr/lib/x86_64-linux-gnu/openblas-pthread",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/local/lib",
        "/opt/homebrew/opt/openblas/lib",
    ];
    for path in &search_paths {
        if std::path::Path::new(path).exists() {
            println!("cargo:rustc-link-search=native={}", path);
        }
    }
    println!("cargo:rustc-link-lib=openblas");
}
