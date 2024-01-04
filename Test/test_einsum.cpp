#include "tensor.h"
#include <vector>
#include <iostream>

using namespace std;

int main()
{

    ts::Tensor<int> t1 = ts::arange<int>(0, 60);
    cout << "t1:" << t1 << endl;

    ts::Tensor<int> temp1 = ts::Tensor<int>({3, 4, 5});
    cout << "temp1:" << temp1 << endl;

    temp1 = t1.reshape({3, 4, 5});

    cout << "temp1 reshaped:" << temp1 << endl;

    // ts::Tensor<int> t2 = ts::arange<int>(0, 60).reshape({3, 5, 4});
    // cout << "t1:\n"
    //      << t1 << endl;
    // cout << "t2:\n"
    //      << t2 << endl;

    // ts::Tensor<int> t3 = ts::einsum<int>("ijk,ikl->ij", {t1, t2});

    std::vector<int> shape1 = {3, 1};
    std::vector<int> shape2 = {3, 4};

    int data1[3] = {1, 2, 3};
    ts::Tensor<int> tensor1 = ts::Tensor<int>(data1, shape1);

    int data2[12] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    ts::Tensor<int> tensor2 = ts::Tensor<int>(data2, shape2);

    std::cout << "tensor1:\n"
              << tensor1 << std::endl;
    std::cout << "tensor2:\n"
              << tensor2 << std::endl;

    std::string equation = "ij,jk->ik";
    ts::Tensor<int> result = ts::einsum<int>(equation, {tensor1, tensor2});

    std::cout << "result:\n"
              << result << std::endl;

    return 0;
}