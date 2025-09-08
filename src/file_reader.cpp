#include "file_reader.h"
#include <exception>
#include <stdexcept>
#include <iostream>

namespace diskann
{

    FileReader::~FileReader()
    {
        if (m_handle != INVALID_HANDLE_VALUE)
        {
            CloseHandle(m_handle);
            m_handle = INVALID_HANDLE_VALUE;
        }
    }

    bool FileReader::Open(const std::string& path)
    {
        m_path = path;
        std::uint32_t creationDisposition = OPEN_EXISTING;

        m_handle = CreateFileA(
            m_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            creationDisposition,
            FILE_FLAG_NO_BUFFERING,
            NULL);

        if (m_handle == INVALID_HANDLE_VALUE)
        {
            return false;
        }

        return true;
    }

    bool FileReader::Read(std::uint64_t offset, std::uint64_t sizeToRead, char* buffer)
    {
        std::uint32_t bytesRead;

        // For FILE_FLAG_NO_BUFFERING, offset and size must be aligned to sector boundaries
        DWORD sectorSize = 4096; // Common sector size, but should ideally get actual sector size
        
        // Ensure offset is aligned to sector boundary
        std::uint64_t alignedOffset = (offset / sectorSize) * sectorSize;
        
        // Ensure size is aligned to sector boundary
        std::uint64_t alignedSize = ((sizeToRead + sectorSize - 1) / sectorSize) * sectorSize;

        OVERLAPPED overlapped = {0};
        overlapped.Offset = static_cast<DWORD>(alignedOffset & 0xFFFFFFFF);
        overlapped.OffsetHigh = static_cast<DWORD>(alignedOffset >> 32);
        overlapped.hEvent = nullptr;

        auto readSuccess = ReadFile(m_handle, buffer, static_cast<DWORD>(alignedSize), reinterpret_cast<LPDWORD>(&bytesRead), &overlapped);

        if (!readSuccess)
        {
            DWORD error = GetLastError();
            std::cout << "ReadFile failed with error: " << error << std::endl;
            return false;
        }

        return true;
    }

    std::uint64_t FileReader::GetFileSize()
    {
                if (m_handle == INVALID_HANDLE_VALUE)
        {
            throw std::runtime_error("File not opened");
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(m_handle, &fileSize))
        {
            throw std::runtime_error("GetFileSizeEx failed");
        }

        return static_cast<std::uint64_t>(fileSize.QuadPart);
    }
}