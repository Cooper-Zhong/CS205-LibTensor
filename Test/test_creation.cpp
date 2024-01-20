#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void test_creation()
{
    // 1.given data
    int *data = new int[10];
    for (int i = 0; i < 10; i++)
    {
        data[i] = i;
    }
    ts::Tensor<int> t0 = ts::Tensor<int>(data, {2, 5});
    cout << t0 << endl;

    // 2.given shape with random data
    ts::Tensor<double> t1 = ts::Tensor<double>::rand_tensor({2, 5});
    cout << t1 << endl;

    // 3. a given shape and data type, and initialize it with a given value
    ts::Tensor<double> t2 = ts::Tensor<double>::zeros_tensor({2, 3});
    cout << t2 << endl;
    ts::Tensor<double> t3 = ts::Tensor<double>::ones_tensor({2, 3});
    cout << t3 << endl;
    ts::Tensor<double> t4 = ts::Tensor<double>::full_tensor({2, 3}, 3.14);
    cout << t4 << endl;

    //4.a given shape and data type, and initialize it to a specific pattern
    ts::Tensor<double> t5 = ts::Tensor<double>::eye_tensor({3,4});
    cout << t5 << endl;

}

int main()
{
    test_creation();
}