
#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void test_transpose()
{
    // test transpose
    ts::Tensor<double> t1 = ts::arange<double>(0, 15).reshape({3, 5});
    cout << t1 << endl;
    ts::Tensor<double> t2 = t1.transpose(0, 1);
    cout << t2 << endl;
    // cout << t1.transpose(0, 1) << endl;

    //permute
    ts::Tensor<double> t3 = t1.permute({1, 0});
    cout << t3 << endl;
}

int main()
{
    test_transpose();
}