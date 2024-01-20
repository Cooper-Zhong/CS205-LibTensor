#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;


void test_permutate()
{
    // test permutate
    vector<int> shape = {2, 2, 2};
    ts::Tensor<double> t = create_test_tensor(shape, true, false);
    cout << t.permute({2, 1, 0}) << endl;
}

int main()
{
    test_permutate();
}