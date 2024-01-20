
#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void test_pointwise()
{
    double *data1 = new double[6];
    data1[0] = 0.1;
    data1[1] = 1.2;
    data1[2] = 2.2;
    data1[3] = 3.1;
    data1[4] = 4.9;
    data1[5] = 5.2;
    double *data2 = new double[6];
    data2[0] = 0.2;
    data2[1] = 1.3;
    data2[2] = 2.3;
    data2[3] = 3.2;
    data2[4] = 4.8;
    data2[5] = 5.1;

    ts::Tensor<double> t1(data1, {2, 3});
    ts::Tensor<double> t2(data2, {2, 3});
    cout << "t1:\n"
         << t1 << endl;
    cout << "t2:\n"
         << t2 << endl;
    ts::Tensor<double> t3 = t1 + t2;
    cout << "t1 + t2:\n" << t3 << endl;
    ts::Tensor<double> t4 = t1 - t2;
    cout << "t1 - t2:\n" << t4 << endl;
    ts::Tensor<double> t5 = t1 * t2;
    cout << "t1 * t2:\n" << t5 << endl;
    ts::Tensor<double> t6 = t1 / t2;
    cout << "t1 / t2:\n" << t6 << endl;
    ts::Tensor<double> t7 = t1.log();
    cout << "t1.log():\n" << t7 << endl;
;
}

int main()
{
    test_pointwise();
}