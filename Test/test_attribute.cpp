#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;


void test_attribute(){
    // test cat
    vector<int> shape_0 = {4, 3, 2};
    vector<int> shape_1 = {4, 3, 2};
    ts::Tensor<double> t0 = ts::arange<double>(0, 20).reshape({4,5});
    ts::Tensor<float> t1 = ts::Tensor<float>(shape_1);
    cout << "t0:\n" << t0 << endl;
    cout << t0.size()<<endl;
    cout << t0.type()<<endl;
    cout << t0.get_data()<<endl;



}

int main(){
    test_attribute();
}