
#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void test_view()
{
    ts::Tensor<double> t1 = ts::arange<double>(0, 15).reshape({3, 5});
    cout << t1 << endl;
    ts::Tensor<double> t2 = t1.view({5, 3});
    cout << t2 << endl;
    // cout << t1.view({5, 3}) << endl;
    ts::Tensor<double> t3 = t1.view({1,15});
    cout << t3 << endl;
    // cout << t1.view({1,15}) << endl;
}

int main()
{
    test_view();
}