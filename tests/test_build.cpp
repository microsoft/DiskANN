//#include <distances.h>
//#include <indexing.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <math_utils.h>
#include <utils.h>

int main(int argc, char** argv) {
  BuildIndex("/mnt/rakri/sift100k.fvecs", "/mnt/rakri/sift100k/sift100k",
             "50 64 750 32");
}
