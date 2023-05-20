#include <boost/test/unit_test.hpp>

#include "memory_manager.h"

BOOST_AUTO_TEST_SUITE(memory_manager_tests)

template <typename T> void test_vector()
{
    diskann::MemoryManager memory_manager;
    BOOST_TEST(memory_manager.get_memory_used_in_bytes() == 0);
    diskann::vector<T> data(memory_manager.create_allocator<T>());
    for (int i = 1; i <= 3; ++i)
    {
        auto pre_size = memory_manager.get_memory_used_in_bytes();
        data.resize(i * 3);
        BOOST_TEST(memory_manager.get_memory_used_in_bytes() > pre_size);
    }

    auto pre_size = memory_manager.get_memory_used_in_bytes();
    data.push_back(1);
    BOOST_TEST(memory_manager.get_memory_used_in_bytes() > pre_size);
}

template <typename T> void test_embedded_vector()
{
    diskann::MemoryManager memory_manager;
    BOOST_TEST(memory_manager.get_memory_used_in_bytes() == 0);
    diskann::vector<diskann::vector<T>> data(memory_manager.create_allocator<T>());
    auto allocator = memory_manager.create_allocator<T>();
    for (int i = 1; i <= 3; ++i)
    {
        auto pre_size = memory_manager.get_memory_used_in_bytes();
        data.resize(i * 10, diskann::vector<T>(allocator));
        BOOST_TEST(memory_manager.get_memory_used_in_bytes() > pre_size);

        for (auto &v : data)
        {
            pre_size = memory_manager.get_memory_used_in_bytes();
            v.resize(v.size() + 1);
            BOOST_TEST(memory_manager.get_memory_used_in_bytes() > pre_size);
        }
    }
}

template <typename T> void test_new_array()
{
    diskann::MemoryManager memory_manager;
    BOOST_TEST(memory_manager.get_memory_used_in_bytes() == 0);
    for (int i = 1; i < 10; ++i)
    {
        auto pre_size = memory_manager.get_memory_used_in_bytes();

        auto ptr = memory_manager.new_array<T>(i);
        BOOST_TEST(ptr != nullptr);
        BOOST_TEST(memory_manager.get_memory_used_in_bytes() == pre_size + sizeof(T) * i);

        memory_manager.delete_array(ptr);
        BOOST_TEST(memory_manager.get_memory_used_in_bytes() == pre_size);
    }
}

template <typename T> void test_basic_data_func()
{
    test_vector<T>();
    test_embedded_vector<T>();
    test_new_array<T>();
}

BOOST_AUTO_TEST_CASE(test_basic_data)
{
    test_basic_data_func<unsigned char>();
    test_basic_data_func<uint8_t>();
    test_basic_data_func<uint16_t>();
    test_basic_data_func<uint32_t>();
    test_basic_data_func<uint64_t>();

    test_basic_data_func<char>();
    test_basic_data_func<int8_t>();
    test_basic_data_func<int16_t>();
    test_basic_data_func<int32_t>();
    test_basic_data_func<int64_t>();

    test_basic_data_func<float>();
    test_basic_data_func<double>();
}

BOOST_AUTO_TEST_CASE(test_allocate)
{
    diskann::MemoryManager memory_manager;
    BOOST_TEST(memory_manager.get_memory_used_in_bytes() == 0);
    for (size_t align = 2; align <= 16; align *= 2)
    {
        for (size_t j = 1; j <= 10; ++j)
        {
            void *ptr = nullptr;
            auto pre_size = memory_manager.get_memory_used_in_bytes();

            memory_manager.alloc_aligned(&ptr, j * align, align);
            BOOST_TEST(ptr != nullptr);
            BOOST_TEST(memory_manager.get_memory_used_in_bytes() == pre_size + j * align);

            memory_manager.aligned_free(ptr);
            BOOST_TEST(memory_manager.get_memory_used_in_bytes() == pre_size);

            memory_manager.alloc_aligned(&ptr, j * align, align);
            BOOST_TEST(ptr != nullptr);
            BOOST_TEST(memory_manager.get_memory_used_in_bytes() == pre_size + j * align);

#ifdef _WINDOWS
            memory_manager.realloc_aligned(&ptr, (j + 1) * align, align);
            BOOST_TEST(ptr != nullptr);
            BOOST_TEST(memory_manager.get_memory_used_in_bytes() == pre_size + (j + 1) * align);

            memory_manager.aligned_free(ptr);
            BOOST_TEST(memory_manager.get_memory_used_in_bytes() == pre_size);
#endif
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
