#ifndef _WINDOWS

#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>


    #include "pm.h"
    #include <memkind.h>


// Must be initialized before use.
// Keep the allocation type of the base allocator to `uint8_t`.
//
// Templated function in the header file can copy-construct allocators with other
// type parameters.
static std::unique_ptr<libmemkind::pmem::allocator<uint8_t>> __allocator;
static std::unordered_map<void*, size_t> __pm_pointers;

const libmemkind::pmem::allocator<uint8_t>& diskann::_pm_allocator()
{
    assert(diskann::is_pm_init());
    return *__allocator.get();
}

bool diskann::is_pm_init()
{
    return bool(__allocator);
}

void diskann::init_pm(const std::string& dir)
{
    if (!is_pm_init()) {
        std::cout << "[PM} Initializing PM: " << dir << std::endl;
        //auto policy = libmemkind::allocation_policy::CONSERVATIVE;
        auto policy = libmemkind::allocation_policy::DEFAULT;
        __allocator = std::make_unique<libmemkind::pmem::allocator<uint8_t>>(dir, 0, policy);
    }
}

void* diskann::__alloc(size_t size, size_t align, bool pm)
{
    void* ptr;
    if (pm) {
        assert(is_pm_init());
        ptr = (void*) __allocator->allocate(size);
        __pm_pointers.insert({ptr, size});
    } else {
        ptr = ::aligned_alloc(align, size);
    }
    return ptr;
}

void diskann::__free(void* ptr)
{
    auto search = __pm_pointers.find(ptr);
    if (search != __pm_pointers.end()) {
        __allocator->deallocate((uint8_t*) ptr, search->second);
        __pm_pointers.erase(search);
    } else {
        free(ptr);
    }
}

#endif
