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
        m_mask = ~static_cast<uint64_t>(m_sectorSize - 1);
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

    std::uint64_t FileReader::Read(std::uint64_t offset, std::uint64_t sizeToRead, char* buffer)
    {
        auto endOffsetToRead = offset + sizeToRead;
        uint64_t beginAddr = offset & m_mask;
        uint64_t endAddr = endOffsetToRead & m_mask;
        if (endAddr != endOffsetToRead)
        {
            endAddr += m_sectorSize;
        }
        uint64_t diskReadSize = endAddr - beginAddr;

        if (diskReadSize > 0)
        {
            auto allocatedBuffer = std::make_unique<uint8_t[]>(diskReadSize);
            void* alignBufferAddress = allocatedBuffer.get();
            size_t bufferSize = 0;

            std::align(m_sectorSize, diskReadSize, alignBufferAddress, bufferSize);

            auto alignedBuffer = (std::uint8_t*)alignBufferAddress;

            if (!ReadData(beginAddr, alignedBuffer, (std::uint32_t)diskReadSize))
            {
                return 0;
            }

            memcpy_s(buffer, sizeToRead, alignedBuffer + offset - beginAddr, sizeToRead);

            return sizeToRead;
        }
        
        return 0;
    }

    bool FileReader::ReadData(std::uint64_t offset, std::uint8_t* buffer, std::uint32_t sizeToRead)
    {
        std::uint32_t bytesRead;

        OVERLAPPED overlapped;
        overlapped.Pointer = (PVOID)offset;
        overlapped.hEvent = nullptr;

        auto readSuccess = ReadFile(m_handle, buffer, sizeToRead, reinterpret_cast<LPDWORD>(&bytesRead), &overlapped);

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