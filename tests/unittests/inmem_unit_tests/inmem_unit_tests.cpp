#include <gtest/gtest.h>
#include "distance.h"
#include "abstract_data_store.h"


//Create a sample unit test 

// Demonstrate some basic assertions.
TEST(InMemTest, ConstructorTest)
{
    auto distance_metric = std::make_shared<diskann::DistanceL2<float>>();
    diskann::AbstractDataStore *data_store = new diskann::InMemDataStore();

    EXPECT_EQ(data_store->get_num_points(), 0);


}
