#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;


void test_rand()
{
    // test permutate
    vector<int> shape = {2, 2, 2};
    ts::Tensor<float> t = ts::Tensor<float>::rand_tensor(shape);
    cout << t << endl;
}

int main()
{
    test_rand();
}