#include "utils_windows.h"

#include <stdexcept>

namespace diskann
{

void utils_windows::write_to_file(const std::string &fileName, const char *data, size_t size)
{
    PagedBinaryWriter writer(fileName, size);

    writer.WriteBytes(data, size);
}


// 
// FileWriterMap 
//
FileWriterMap::FileWriterMap(const std::string &fileName, size_t fileSize)
{
    // Create or open the file
    m_file = CreateFileA(fileName.c_str(), 
                         GENERIC_READ | GENERIC_WRITE, 
                         0, 
                         NULL, 
                         CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, NULL);

    if (m_file == INVALID_HANDLE_VALUE)
    {
        throw std::runtime_error("Failed to create file. Error: " + std::to_string(GetLastError()));
    }

    // Set the file size
    //const LONG fileSizeW = static_cast<LONG>(fileSize);
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(fileSize);
    SetFilePointerEx(m_file, li, NULL, FILE_BEGIN);
    SetEndOfFile(m_file);

    const DWORD sizeLow = static_cast<DWORD>(fileSize & 0xFFFFFFFF);
    const DWORD sizeHigh = static_cast<DWORD>((fileSize >> 32) & 0xFFFFFFFF);
    m_fileMap = CreateFileMappingA(m_file, NULL, PAGE_READWRITE, sizeHigh, sizeLow, NULL);
    if (m_fileMap == NULL)
    {
        CloseHandle(m_file);
        throw std::runtime_error("Failed to create file mapping. Error: " + std::to_string(GetLastError()));
    }
}


FileWriterMap::~FileWriterMap()
{
    if (m_fileMap)
    {
        CloseHandle(m_fileMap);
    }

    if (m_file)
    {
        CloseHandle(m_file);
    }
}


HANDLE& FileWriterMap::GetMap()
{
    return m_fileMap;
}


//
// PagedBinaryWriter
//
static const size_t PAGE_SIZE = 256 * 1024 * 1024; // 256 MB

// Periodically flush data to disk after writing this many bytes to avoid memory writing outpacing flushing to disk
// and leading to temporary memory spikes.
static const size_t FLUSH_SIZE = 512 * 1024 * 1024; // 512 MB

// Ensure FLUSH_SIZE is a multiple of PAGE_SIZE for simplicity.
static_assert(FLUSH_SIZE % PAGE_SIZE == 0, "FLUSH_SIZE must be a multiple of PAGE_SIZE");

namespace
{
    // Align the requested page size to the system's allocation granularity.
    size_t AlignPageSize(size_t requestedPageSize)
    {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return (requestedPageSize + sysInfo.dwAllocationGranularity - 1) & ~(sysInfo.dwAllocationGranularity - 1);
    }
}

PagedBinaryWriter::PagedBinaryWriter(const std::string& fileName, size_t fileSize)
    : m_file(fileName, fileSize), 
      m_position(0), 
      m_pageOffset(0), 
      m_pageView(nullptr), 
      m_pageSize(AlignPageSize(PAGE_SIZE)),
      m_maxFileSize(fileSize),
      m_nextFlushPosition(FLUSH_SIZE)
{
    OpenNextPage();
}


PagedBinaryWriter::~PagedBinaryWriter()
{
    CloseCurrentPage();
}


void PagedBinaryWriter::WriteBytes(const void *data, size_t size)
{
    size_t bytesToWrite = size;

    while (bytesToWrite > 0)
    {
        size_t remainingInPage = m_pageSize - m_pageOffset;
        if (remainingInPage == 0)
        {
            OpenNextPage();
            remainingInPage = m_pageSize;
        }

        size_t toWrite = std::min(bytesToWrite, remainingInPage);

        std::memcpy(static_cast<char *>(m_pageView) + m_pageOffset, data, toWrite);

        // Advance offsets and the data pointer.
        m_pageOffset += toWrite;
        m_position += toWrite;
        bytesToWrite -= toWrite;
        data = static_cast<const char *>(data) + toWrite;
    }
}


void PagedBinaryWriter::Rewind(size_t position)
{
    // Assume the current page has finished writing.
    CloseCurrentPage();

    // Open the page that contains the specified position.
    m_position = position;
    OpenNextPage();
}



void PagedBinaryWriter::CloseCurrentPage()
{
    if (m_pageView)
    {
        if (m_position >= m_nextFlushPosition)
        {
            // Flush the view to disk.
            if (!FlushViewOfFile(m_pageView, 0))
            {
                throw std::runtime_error("Failed to flush view of file. Error: " + std::to_string(GetLastError()));
            }

            FlushFileBuffers(m_file.GetMap());

            m_nextFlushPosition = m_position + FLUSH_SIZE;
        }

        UnmapViewOfFile(m_pageView);
        m_pageView = nullptr;
    }
}


void PagedBinaryWriter::OpenNextPage()
{
    CloseCurrentPage();

    if (m_position >= m_maxFileSize)
    {
        throw std::runtime_error("Attempt to write beyond the maximum file size.");
    }

    DWORD offsetLow = static_cast<DWORD>(m_position & 0xFFFFFFFF);
    DWORD offsetHigh = static_cast<DWORD>((m_position >> 32) & 0xFFFFFFFF);

    size_t remainingSize = m_maxFileSize - m_position;
    remainingSize = std::min(remainingSize, m_pageSize);

    m_pageView = MapViewOfFile(m_file.GetMap(), FILE_MAP_WRITE, offsetHigh, offsetLow, remainingSize);
    m_pageOffset = 0;

    if (!m_pageView)
    {
        throw std::runtime_error("Failed to map view at offset: " + std::to_string(m_position) +
                                 ", requested size: " + std::to_string(remainingSize) + 
                                 ", total file size: " + std::to_string(m_maxFileSize) +
                                 ". Error: " + std::to_string(GetLastError()));
    }
}

} // namespace diskann