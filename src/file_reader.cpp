#include "file_reader.h"
#include <exception>
#include <stdexcept>

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
            GENERIC_READ | GENERIC_WRITE,
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

        OVERLAPPED overlapped;
        overlapped.Pointer = (PVOID)offset;
        overlapped.hEvent = nullptr;

        auto readSuccess = ReadFile(m_handle, buffer, (std::uint32_t)sizeToRead, reinterpret_cast<LPDWORD>(&bytesRead), &overlapped);

        if (!readSuccess)
        {
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