#pragma once

#include <map>


#include "file_content.h"


#ifdef EXEC_ENV_OLS
namespace diskann {
  class MemoryMappedFiles {
   public:
    void addFile(const std::string& fileName, FileContent fc);
    std::string& filesPrefix();     // Should be called only after all the fileblobs have been added.
    bool fileExists(const std::string& fileName);
    FileContent& getContent(const std::string& fileName);


   private:

    // This function assumes that the two files have the first character in
    // common. This is true for the files created by diskann::build_disk_index.
    std::string lcsAtZero(const std::string& s1, const std::string& s2);
    void manageCommonPrefix(const std::string& newFileName);

    std::map<const std::string, FileContent> _nameContentMap;
    std::string                              _filesPrefix;
  };

}  // namespace diskann
#endif