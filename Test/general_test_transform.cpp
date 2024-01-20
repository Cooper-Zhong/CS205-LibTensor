#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void general_test_transform(){
    ts::Tensor<double> t = ts::Tensor<double>({3, 5}, {0.1, 1.2, 3.4, 5.6, 7.8, 2.2, 3.1, 4.5, 6.7, 8.9, 4.9, 5.2, 6.3, 7.4, 8.5});
    ts::Tensor<double> t1 = ts::arange<double>(0, 20).reshape({4,5});
    ts::Tensor<double> t2 = ts::arange<double>(20, 40).reshape({4,5});
    cout << "t1:\n" << t1 << endl;
    cout << "t2:\n" << t2 << endl;

    cout << "-------------- Indexing Test --------------" << endl;
    cout << "t1[0]:\n" << t1[0] << endl;

}

int main(){
    general_test_transform();
}