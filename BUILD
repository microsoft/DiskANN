# BUILD

cc_library(
    name = "diskann_library",
    srcs = glob(["src/*.cpp"]),
    hdrs = glob(["include/*.h", "include/tsl/*.h"]),
    includes = ["include", "include/tsl"],
    copts = [
        "-ltcmalloc", "-Ofast", "-march=native", "-mtune=native",
        "-I/usr/include/mkl", "-m64", "-Wl,--no-as-needed",
        "-DMKL_ILP64", "-DNDEBUG", "-DUSE_AVX2",
        "-mavx2", "-mfma", "-msse2",
        "-ftree-vectorize", "-fno-builtin-malloc", "-fno-builtin-calloc", "-fno-builtin-realloc",
        "-fno-builtin-free", "-fopenmp", "-fopenmp-simd",
        "-funroll-loops", '-Wfatal-errors'
    ],
    visibility = ["//visibility:public"],
)
