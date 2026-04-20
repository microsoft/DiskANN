There are several things being tested here.

First, we check that the "test-input-dim-tolerance" can match multiple instances of the
"test-input-dim" input.

Next, we have two tests for "test-input-types" - one of which is more specific than the other.
This tests the situation where multiple regression tests have the same input and tolerance file types, but we still need to run matching on the input to find the best one.

Finally, this tests the printing and output generation of success.
