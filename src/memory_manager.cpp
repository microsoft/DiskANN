// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "memory_manager.h"
#include "utils.h"

namespace diskann
{

MemoryManager &MemoryManager::get_instance()
{
    static MemoryManager instance;
    return instance;
}

size_t MemoryManager::get_memory_used_in_bytes() const
{
    return _memory_used_in_bytes;
}

void MemoryManager::alloc_aligned(void **ptr, size_t size, size_t align)
{
    *ptr = nullptr;
    if (IS_ALIGNED(size, align) == 0)
        report_misalignment_of_requested_size(size, align);
#ifndef _WINDOWS
    *ptr = ::aligned_alloc(align, size);
#else
    *ptr = ::_aligned_malloc(size, align); // note the swapped arguments!
#endif
    if (*ptr == nullptr)
    {
        report_memory_allocation_failure(size, align);
        return;
    }

    insert_block(*ptr, size);
}

void MemoryManager::realloc_aligned(void **ptr, size_t size, size_t align)
{
    if (IS_ALIGNED(size, align) == 0)
        report_misalignment_of_requested_size(size, align);
#ifdef _WINDOWS
    erase_block(*ptr);
    *ptr = ::_aligned_realloc(*ptr, size, align);
#else
    diskann::cerr << "No aligned realloc on GCC. Must malloc and mem_align, "
                     "left it out for now."
                  << std::endl;
#endif
    if (*ptr == nullptr)
        report_memory_allocation_failure(size, align);

    insert_block(*ptr, size);
}

void MemoryManager::aligned_free(void *ptr)
{
    // Gopal. Must have a check here if the pointer was actually allocated by
    // _alloc_aligned
    if (ptr == nullptr)
    {
        return;
    }
#ifndef _WINDOWS
    free(ptr);
#else
    ::_aligned_free(ptr);
#endif

    erase_block(ptr);
}

void MemoryManager::insert_block(void *ptr, size_t size)
{
    if (ptr == nullptr)
    {
        diskann::cerr << "block address should not be null" << std::endl;
        return;
    }

    auto it = _block_to_size.find(ptr);
    if (it != _block_to_size.end())
    {
        diskann::cerr << "block address " << ptr << " conflicts" << std::endl;
        return;
    }

    _block_to_size[ptr] = size;
    _memory_used_in_bytes += size;
}

void MemoryManager::erase_block(void *ptr)
{
    if (ptr == nullptr)
    {
        diskann::cerr << "block address should not be null" << std::endl;
        return;
    }

    auto it = _block_to_size.find(ptr);
    if (it == _block_to_size.end())
    {
        diskann::cerr << "can not find block address " << ptr << " in _block_to_size" << std::endl;
        return;
    }

    auto size = it->second;
    _block_to_size.unsafe_erase(ptr);
    _memory_used_in_bytes -= size;
}

template <typename T> T *MemoryManager::new_array(size_t array_size)
{
    auto ret = new T[array_size];
    auto size = sizeof(T) * array_size;
    insert_block(ret, size);
    return ret;
}

template <typename T> void MemoryManager::delete_array(T *ptr)
{
    erase_block(ptr);
    delete[] ptr;
}

template <typename T> Allocator<T> MemoryManager::create_allocator()
{
    return Allocator<T>(&_memory_used_in_bytes);
}

template DISKANN_DLLEXPORT char *MemoryManager::new_array(size_t);
template DISKANN_DLLEXPORT int8_t *MemoryManager::new_array(size_t);
template DISKANN_DLLEXPORT uint8_t *MemoryManager::new_array(size_t);
template DISKANN_DLLEXPORT uint32_t *MemoryManager::new_array(size_t);
template DISKANN_DLLEXPORT int32_t *MemoryManager::new_array(size_t);
template DISKANN_DLLEXPORT uint64_t *MemoryManager::new_array(size_t);
template DISKANN_DLLEXPORT int64_t *MemoryManager::new_array(size_t);
template DISKANN_DLLEXPORT float *MemoryManager::new_array(size_t);

template DISKANN_DLLEXPORT void MemoryManager::delete_array(char *);
template DISKANN_DLLEXPORT void MemoryManager::delete_array(int8_t *);
template DISKANN_DLLEXPORT void MemoryManager::delete_array(uint8_t *);
template DISKANN_DLLEXPORT void MemoryManager::delete_array(uint32_t *);
template DISKANN_DLLEXPORT void MemoryManager::delete_array(int32_t *);
template DISKANN_DLLEXPORT void MemoryManager::delete_array(uint64_t *);
template DISKANN_DLLEXPORT void MemoryManager::delete_array(int64_t *);
template DISKANN_DLLEXPORT void MemoryManager::delete_array(float *);

template Allocator<uint32_t> MemoryManager::create_allocator();
template Allocator<int> MemoryManager::create_allocator();
template Allocator<diskann::vector<uint32_t>> MemoryManager::create_allocator();

} // namespace diskann
