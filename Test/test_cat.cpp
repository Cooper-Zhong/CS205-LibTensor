#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;


void test_cat(){
    // test cat
    vector<int> shape_0 = {2, 3};
    vector<int> shape_1 = {2, 3};
    ts::Tensor<double> t0 = create_test_tensor(shape_0, false, false);
    ts::Tensor<double> t1 = create_test_tensor(shape_1, false, false);
    ts::Tensor<double> t2 = t0.cat(t1, 0);
    ts::Tensor<double> t3 = t0.cat(t1, 1);
    cout << "cat 0:\n" << t0 << endl;
    cout << "cat 1:\n" << t1 << endl;
    cout << "cat 2:\n" << t2 << endl;
    cout << "cat 3:\n" << t3 << endl;
}

int main(){
    test_cat();
}