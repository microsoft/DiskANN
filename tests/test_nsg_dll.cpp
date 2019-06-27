#include <dll/nsg_interface.h>
#include "dll/IANNIndex.h"

int main(int argc, char* argv[]) {
  // argv[1]: data file
  // argv[2]: output_file_pattern
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <data_file> <output_file_prefix>"
              << std::endl;
  }

  ANNIndex::IANNIndex* intf = new NSG::NSGInterface<float>(0, ANNIndex::DT_L2);
  intf->BuildIndex(argv[1], argv[2], "50 64 500 32");
}