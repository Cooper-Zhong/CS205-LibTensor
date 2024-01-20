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
    cout << "t1 before mutating: " << endl << t1;
    t1[{1,2}]=77;
    cout << "t1[{1,2}]=77" << endl;
    cout << "t1 after mutating: " << endl << t1;


    t1[{1,2}]=7;

    cout << "--------------- Transpose Test ---------------" << endl;
    cout << "t1 before transposing " << endl << t1;
    cout << "t1 data pointer: " << t1.get_data() << endl;
    test = t1.transpose(0,1);
    cout << "t1 after transposing " << endl << test;
    cout << "t1 data pointer: " << test.get_data() << endl;

    cout << "--------------- Permute Test ---------------" << endl;
    cout << "t1 before permuting " << endl << t1;
    cout << "t1 data pointer: " << t1.get_data() << endl;
    test = t1.permute({1,0});
    cout << "t1 after permuting " << endl << test;
    cout << "t1 data pointer: " << test.get_data() << endl;

    cout << "--------------- View Test ---------------" << endl;
    cout << "t1 before permuting " << endl << t1;
    cout << "t1 data pointer: " << t1.get_data() << endl;
    test = t1.view({2,5,2});
    cout << "t1 after permuting " << endl << test;
    cout << "t1 data pointer: " << test.get_data() << endl;





}

int main(){
    general_test_transform();

}