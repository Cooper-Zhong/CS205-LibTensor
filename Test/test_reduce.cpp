#include "tensor.h"
#include <vector>
#include <iostream>

using namespace std;

int main()
{
    double *data1 = new double[6];
    data1[0] = 0.1;
    data1[1] = 1.2;
    data1[2] = 2.2;
    data1[3] = 3.1;
    data1[4] = 4.9;
    data1[5] = 5.2;
    ts::Tensor<double> t1(data1, {2, 3});
    cout << "t1:\n"
         << t1 << endl;
    // sum
    ts::Tensor<double> t2 = t1.sum(1);
    cout << "t1.sum(1):\n"
         << t2 << endl;
    ts::Tensor<double> t3 = t1.sum(0);
    cout << "t1.sum(0):\n"
         << t3 << endl;

    // mean
    ts::Tensor<double> t4 = t1.mean(1);
    cout << "t1.mean(1):\n"
         << t4 << endl;
    // max
    ts::Tensor<double> t5 = t1.max(1);
    cout << "t1.max(1):\n"
         << t5 << endl;
    // min
    ts::Tensor<double> t6 = t1.min(1);
    cout << "t1.min(1):\n"
         << t6 << endl;
}