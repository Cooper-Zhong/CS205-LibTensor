#include <vector>
#include <iostream>

#include "tensor.h"
#include "test.h"


void test_reshape(){
    // test cat
    vector<int> shape_0 = {1, 60};
    vector<int> shape_1 = {4, 3, 2};
    ts::Tensor<double> t0 = create_test_tensor(shape_0, false, false);
    ts::Tensor<double> t1 = t0.contiguous();
    ts::Tensor<double> t2 = t1.reshape({3, 4, 5});


    cout << "t0:\n" << t0 << endl;
    cout << "t1:\n" << t1 << endl;
    cout << "t2:\n" << t2 << endl;
}

int main(){
    test_reshape();
}