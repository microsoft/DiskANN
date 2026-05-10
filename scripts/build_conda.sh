#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the conda environment first (e.g., conda activate diskann-conda)." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${ROOT_DIR}/.deps"
SRC_DIR="${DEPS_DIR}/src"
PREFIX_DIR="${DEPS_DIR}/prefix"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${SRC_DIR}" "${PREFIX_DIR}"

JOBS="${JOBS:-}"
if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  fi
fi

if command -v mamba >/dev/null 2>&1; then
  INSTALLER="mamba"
else
  INSTALLER="conda"
fi

conda_install() {
  local pkgs=("$@")
  "${INSTALLER}" install -y -c conda-forge -c anaconda -c https://software.repos.intel.com/python/conda/ "${pkgs[@]}"
}

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "${out}" "${url}"
  else
    wget -O "${out}" "${url}"
  fi
}

ensure_conda_deps() {
  if ! conda_install boost-cpp gperftools libaio libunwind mkl-devel intel-openmp cmake make pkg-config binutils_linux-64 gcc_linux-64 gxx_linux-64; then
    echo "Conda install failed; will attempt source builds for Boost, gperftools, and libaio." >&2
  fi
}

ensure_mkl() {
  local mkl_lib="${MKL_PATH:-${CONDA_PREFIX}/lib}/libmkl_core.so"
  local mkl_inc="${MKL_INCLUDE_PATH:-${CONDA_PREFIX}/include}/mkl.h"
  local omp_lib="${OMP_PATH:-${CONDA_PREFIX}/lib}/libiomp5.so"

  if [[ ! -f "${mkl_lib}" || ! -f "${mkl_inc}" || ! -f "${omp_lib}" ]]; then
    echo "MKL or Intel OpenMP not found; retrying conda install from Intel channel." >&2
    conda_install mkl-devel intel-openmp || true
  fi

  if [[ ! -f "${mkl_lib}" || ! -f "${mkl_inc}" || ! -f "${omp_lib}" ]]; then
    echo "MKL/Intel OpenMP still missing. Install them via conda (mkl-devel, intel-openmp) and retry." >&2
    exit 1
  fi
}

ensure_boost() {
  local have_boost=0
  if ls "${CONDA_PREFIX}"/lib/libboost_program_options* >/dev/null 2>&1; then
    have_boost=1
  elif ls "${PREFIX_DIR}"/lib/libboost_program_options* >/dev/null 2>&1; then
    have_boost=1
  fi

  if [[ "${have_boost}" -eq 1 ]]; then
    return 0
  fi

  local boost_version="1.85.0"
  local boost_version_underscore="1_85_0"
  local boost_tar="boost_${boost_version_underscore}.tar.gz"
  local boost_url="https://boostorg.jfrog.io/artifactory/main/release/${boost_version}/source/${boost_tar}"
  local boost_src="${SRC_DIR}/boost_${boost_version_underscore}"

  if [[ ! -d "${boost_src}" ]]; then
    echo "Building Boost from source (${boost_version})..." >&2
    download "${boost_url}" "${SRC_DIR}/${boost_tar}"
    tar -xzf "${SRC_DIR}/${boost_tar}" -C "${SRC_DIR}"
  fi

  pushd "${boost_src}" >/dev/null
  ./bootstrap.sh --with-libraries=program_options --prefix="${PREFIX_DIR}"
  ./b2 install -j "${JOBS}"
  popd >/dev/null
}

ensure_gperftools() {
  if ls "${CONDA_PREFIX}"/lib/libtcmalloc* >/dev/null 2>&1; then
    return 0
  fi
  if ls "${PREFIX_DIR}"/lib/libtcmalloc* >/dev/null 2>&1; then
    return 0
  fi

  local gpt_version="2.17.2"
  local gpt_tar="gperftools-${gpt_version}.tar.gz"
  local gpt_url="https://github.com/gperftools/gperftools/releases/download/gperftools-${gpt_version}/${gpt_tar}"
  local gpt_src="${SRC_DIR}/gperftools-${gpt_version}"

  if [[ ! -d "${gpt_src}" ]]; then
    echo "Building gperftools from source (${gpt_version})..." >&2
    download "${gpt_url}" "${SRC_DIR}/${gpt_tar}"
    tar -xzf "${SRC_DIR}/${gpt_tar}" -C "${SRC_DIR}"
  fi

  pushd "${gpt_src}" >/dev/null
  ./configure --prefix="${PREFIX_DIR}"
  make -j "${JOBS}"
  make install
  popd >/dev/null
}

ensure_libaio() {
  if [[ -f "${CONDA_PREFIX}/lib/libaio.so" || -f "${PREFIX_DIR}/lib/libaio.so" ]]; then
    return 0
  fi

  local libaio_version="0.3.113"
  local libaio_tar="libaio-${libaio_version}.tar.gz"
  local libaio_url="https://pagure.io/libaio/archive/libaio-${libaio_version}/${libaio_tar}"
  local libaio_src="${SRC_DIR}/libaio-${libaio_version}"

  if [[ ! -d "${libaio_src}" ]]; then
    echo "Building libaio from source (${libaio_version})..." >&2
    download "${libaio_url}" "${SRC_DIR}/${libaio_tar}"
    tar -xzf "${SRC_DIR}/${libaio_tar}" -C "${SRC_DIR}"
  fi

  pushd "${libaio_src}" >/dev/null
  make -j "${JOBS}"
  make install prefix="${PREFIX_DIR}" libdir="${PREFIX_DIR}/lib"
  popd >/dev/null
}

ensure_conda_deps
ensure_mkl
ensure_boost
ensure_gperftools
ensure_libaio

export CMAKE_PREFIX_PATH="${PREFIX_DIR}:${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CPATH="${PREFIX_DIR}/include:${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${PREFIX_DIR}/lib:${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${PREFIX_DIR}/lib:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PKG_CONFIG_PATH="${PREFIX_DIR}/lib/pkgconfig:${CONDA_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

MKL_PATH="${MKL_PATH:-${CONDA_PREFIX}/lib}"
MKL_INCLUDE_PATH="${MKL_INCLUDE_PATH:-${CONDA_PREFIX}/include}"
OMP_PATH="${OMP_PATH:-${CONDA_PREFIX}/lib}"

REAL_CC_BIN="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
REAL_CXX_BIN="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
LD_BIN="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ld"
AR_BIN="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ar"
RANLIB_BIN="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ranlib"

if [[ ! -x "${REAL_CC_BIN}" ]]; then
  REAL_CC_BIN="${CONDA_PREFIX}/bin/gcc"
fi
if [[ ! -x "${REAL_CXX_BIN}" ]]; then
  REAL_CXX_BIN="${CONDA_PREFIX}/bin/g++"
fi
if [[ ! -x "${LD_BIN}" ]]; then
  LD_BIN="${CONDA_PREFIX}/bin/ld"
fi
if [[ ! -x "${AR_BIN}" ]]; then
  AR_BIN="${CONDA_PREFIX}/bin/ar"
fi
if [[ ! -x "${RANLIB_BIN}" ]]; then
  RANLIB_BIN="${CONDA_PREFIX}/bin/ranlib"
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  rm -rf "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/CMakeFiles"
fi

TOOLCHAIN_BIN="${DEPS_DIR}/toolchain/bin"
mkdir -p "${TOOLCHAIN_BIN}"

if [[ -x "${LD_BIN}" ]]; then
  cat > "${TOOLCHAIN_BIN}/ld" <<EOF
#!/usr/bin/env bash
exec "${LD_BIN}" "\$@"
EOF
  chmod +x "${TOOLCHAIN_BIN}/ld"
fi

if [[ -x "${REAL_CC_BIN}" ]]; then
  cat > "${TOOLCHAIN_BIN}/gcc" <<EOF
#!/usr/bin/env bash
exec "${REAL_CC_BIN}" "\$@"
EOF
  chmod +x "${TOOLCHAIN_BIN}/gcc"
  cat > "${TOOLCHAIN_BIN}/cc" <<EOF
#!/usr/bin/env bash
exec "${REAL_CC_BIN}" "\$@"
EOF
  chmod +x "${TOOLCHAIN_BIN}/cc"
fi

if [[ -x "${REAL_CXX_BIN}" ]]; then
  cat > "${TOOLCHAIN_BIN}/g++" <<EOF
#!/usr/bin/env bash
exec "${REAL_CXX_BIN}" "\$@"
EOF
  chmod +x "${TOOLCHAIN_BIN}/g++"
  cat > "${TOOLCHAIN_BIN}/c++" <<EOF
#!/usr/bin/env bash
exec "${REAL_CXX_BIN}" "\$@"
EOF
  chmod +x "${TOOLCHAIN_BIN}/c++"
fi

export PATH="${TOOLCHAIN_BIN}:${CONDA_PREFIX}/bin:${PATH}"
CC_BIN="${TOOLCHAIN_BIN}/gcc"
CXX_BIN="${TOOLCHAIN_BIN}/g++"
export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export LD="${LD_BIN}"
export AR="${AR_BIN}"
export RANLIB="${RANLIB_BIN}"

BFLAG="-B${TOOLCHAIN_BIN}"
export CFLAGS="${BFLAG}${CFLAGS:+ ${CFLAGS}}"
export CXXFLAGS="${BFLAG}${CXXFLAGS:+ ${CXXFLAGS}}"
export LDFLAGS="${BFLAG}${LDFLAGS:+ ${LDFLAGS}}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC_BIN}" \
  -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
  -DCMAKE_LINKER="${LD_BIN}" \
  -DCMAKE_AR="${AR_BIN}" \
  -DCMAKE_RANLIB="${RANLIB_BIN}" \
  -DCMAKE_C_FLAGS="${BFLAG}" \
  -DCMAKE_CXX_FLAGS="${BFLAG}" \
  -DCMAKE_EXE_LINKER_FLAGS="${BFLAG}" \
  -DCMAKE_SHARED_LINKER_FLAGS="${BFLAG}" \
  -DCMAKE_MODULE_LINKER_FLAGS="${BFLAG}" \
  -DMKL_PATH="${MKL_PATH}" \
  -DMKL_INCLUDE_PATH="${MKL_INCLUDE_PATH}" \
  -DOMP_PATH="${OMP_PATH}"

cmake --build "${BUILD_DIR}" -j "${JOBS}"
