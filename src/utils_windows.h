#pragma once

#include <stdexcept>
#include <string>
#include <windows.h>

namespace diskann
{
class utils_windows
{
public:
    // Function to write the binary data to a file using the file mapping API.
    static void write_to_file(const std::string &fileName, const char *data, size_t size);
};


// Class that provides a memory-mapped file for writing data.
class FileWriterMap
{
public:
    // Creates a map view of the file.
    FileWriterMap(const std::string& fileName, size_t fileSize);

    ~FileWriterMap();

    HANDLE& GetMap();

private:
    // Handle to the file.
    HANDLE m_file;

    // Handle to the file mapping object.
    HANDLE m_fileMap;
};


// Class that allows to write binary data to a buffer in a sequential manner.
class BinaryWriter
{
public:
    BinaryWriter(char *buffer) : ptr(buffer)
    {
    }

    template<typename T>
    void write(T value)
    {
        std::memcpy(ptr, &value, sizeof(value));
        ptr += sizeof(value);
    }

    void write(const void *data, size_t size)
    {
        std::memcpy(ptr, data, size);
        ptr += size;
    }

    char *current() const
    {
        return ptr;
    }

    void reset(char *buffer)
    {
        ptr = buffer;
    }

private:
    char *ptr;
};

// PagedBinaryWriter is a class to write binary data to a file using map view of file in Windows.
// It is most efficient for writing of large binary data to a file.
class PagedBinaryWriter
{
public:
    PagedBinaryWriter(const std::string& filename, size_t fileSize);

    ~PagedBinaryWriter();

    // Writes out bytes of data.
    void WriteBytes(const void *data, size_t size);

    // Rewinds the writer to a specific position in the file.
    void Rewind(size_t position);

    // Writes a value of type T to the file.
    template <typename T> void Write(T value)
    {
        WriteBytes(&value, sizeof(value));
    }

 private:
    // File handle and the file mapping object.
    FileWriterMap m_file;

    // Global position in the file where current writing occurs.
    size_t m_position;

    // Pointer to the current page in the file.
    LPVOID m_pageView;

    // Offset in the current page where the next write will occur.
    size_t m_pageOffset;

    // Page size aligned to the systems allocation size.
    const size_t m_pageSize;

    const size_t m_maxFileSize;

    void CloseCurrentPage();

    void OpenNextPage();
};


} // namespace diskann