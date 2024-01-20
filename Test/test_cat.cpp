#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;


void test_cat(){
    // test cat
    vector<int> shape_0 = {3, 4};
    vector<int> shape_1 = {4, 4};
    vector<int> shape_2 = {5, 4};
    ts::Tensor<double> t0 = ts::Tensor<double>::rand_tensor(shape_0);
    ts::Tensor<double> t1 = ts::Tensor<double>::rand_tensor(shape_1);
    ts::Tensor<double> t2 = ts::Tensor<double>::rand_tensor(shape_2);
    ts::Tensor<double> t3 = t0.cat(t1, 0).cat(t2, 0);

    cout << "cat 0:\n" << t0 << endl;
    cout << "cat 1:\n" << t1 << endl;
    cout << "cat 2:\n" << t2 << endl;
    cout << "cat 3:\n" << t3 << endl;
}

int main(){
    test_cat();
}