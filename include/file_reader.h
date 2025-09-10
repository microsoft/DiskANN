#pragma once
#ifdef _WINDOWS
#include <Windows.h>
#include <string>
#include <memory>

namespace diskann
{

class FileReader
{
public:
    FileReader() = default;

    ~FileReader();

    bool Open(const std::string &path);

    std::uint64_t Read(std::uint64_t offset, std::uint64_t sizeToRead, char* buffer);

    bool ReadData(std::uint64_t offset, std::uint8_t* buffer, std::uint32_t bufferSize);

    std::uint64_t GetFileSize();

private:
    HANDLE m_handle = INVALID_HANDLE_VALUE;
    std::string m_path;
    std::uint64_t m_mask;
    std::uint32_t m_sectorSize = c_defaultSectorSize;
    const static uint16_t c_defaultSectorSize = 512;
};
}

#endif