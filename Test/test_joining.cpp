
#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void test_attribute()
{
    ts::Tensor<double> t1 = ts::arange<double>(0, 6).reshape({2, 3});
    ts::Tensor<double> t2 = ts::arange<double>(0, 12, 2).reshape({2, 3});
    cout << t1 << endl;
    cout << t2 << endl;
    ts::Tensor<double> t3 = t1.cat(t2, 0);
    cout << t3 << endl;
    ts::Tensor<double> t4 = t1.cat(t2, 1);
    cout << t4 << endl;
    ts::Tensor<double> t5 = t1.tile({2, 2});
    cout << t5 << endl;
    // cout << t1.tile({2, 2}) << endl;
}

int main()
{
    test_attribute();
}