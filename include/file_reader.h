#pragma once
#ifdef _WINDOWS
#include <Windows.h>
#include <string>

namespace diskann
{

class FileReader
{
public:
    FileReader() = default;

    ~FileReader();

    bool Open(const std::string &path);

    bool Read(std::uint64_t offset, std::uint64_t sizeToRead, char* buffer);

    std::uint64_t GetFileSize();

private:
    HANDLE m_handle = INVALID_HANDLE_VALUE;
    std::string m_path;

    std::uint32_t m_sectorSize = c_defaultSectorSize;
    const static uint16_t c_defaultSectorSize = 512;
};
}

#endif