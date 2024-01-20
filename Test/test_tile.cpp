#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void test_tile(){
    // test tile
    vector<int> shape = {2, 2};
    ts::Tensor<double> t = create_test_tensor(shape, false, true);
    ts::Tensor<double> t1 = t.tile({4, 3});

    cout << "tile 1:\n" << t1 << endl;
}

int main(){
    test_tile();
}