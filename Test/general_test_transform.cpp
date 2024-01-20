#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>
#include "test.h"

using namespace std;

void general_test_transform(){
    ts::Tensor<double> t1 = ts::arange<double>(0, 20).reshape({4,5});
    ts::Tensor<double> t2 = ts::arange<double>(20, 40).reshape({4,5});
    cout << "t1:\n" << t1 << endl;
    cout << "t2:\n" << t2 << endl;

    ts::Tensor<double> test = ts::Tensor<double>({4,5});

    cout << "--------------- Indexing Test ---------------" << endl;
    test=t1.indexing({1,-1});
    cout << "t1[1]:\n" << test << endl;
    cout << "t1[1] data pointer: " << test.get_data() << endl;
    cout << "t1 data pointer: " << t1.get_data() << endl;

    cout << "--------------- Slicing Test ---------------" << endl;
    test=t1.slicing({{1,3},{2,4}});
    cout << "t1[1:3,2:4]:\n" << test << endl;
    cout << "t1[1:3,2:4] data pointer: " << test.get_data() << endl;
    cout << "t1 data pointer: " << t1.get_data() << endl;

    cout << "--------------- Mutating Test ---------------" << endl;
    t1[]
    cout << "t1[1:3,2:4]:\n" << test << endl;
    cout << "t1[1:3,2:4] data pointer: " << test.get_data() << endl;
    cout << "t1 data pointer: " << t1.get_data() << endl;


}

int main(){
    general_test_transform();

}