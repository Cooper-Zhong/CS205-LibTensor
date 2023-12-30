#include <vector>
#include <iostream>

#include "tensor.h"

using namespace std;

int main() {
//    vector<int> size_0 = {3,4,2};
//    vector<int> size_1 = {4,4,4};
//    double data[24]={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
//    , 17, 18, 19, 20, 21, 22, 23};
//    ts::Tensor<double> t0 = ts::Tensor<double>(data,size_0);
//    ts::Tensor<double> t1 = t0({{1,3},{0,4},{0,1}});
//    ts::Tensor<double> t2 = ts::Tensor<double>(data,{2,4,1});
//
//    cout << "t0:\n" << t0;
//    cout << "t1:\n" << t1;
//    cout << "t2:\n" << t2;
//
//    t2 = t1;
//
//    cout << "t1:\n" << t1;
//    cout << "t2:\n" << t2;

    // cout << t.get_type() << endl;
    // cout << t.get_data() << endl;
    // auto t_data=t.get_data();
    // for (int i = 0; i < t.get_data_length(); i++)
    // {
    //     cout << t_data[i] << " ";
    // }
    // cout << endl;

    // cout << t.get_offset() << endl;
    // cout << t.get_ndim() << endl;
    // cout << t.get_data_length() << endl;

    // auto t_shape = t.get_shape();
    // auto t_stride = t.get_stride();

    // for (int i = 0; i < t_shape.size(); i++)
    // {
    //     cout << t_shape[i] << " ";
    // }
    // cout << endl;

    // for (int i = 0; i < t_stride.size(); i++)
    // {
    //     cout << t_stride[i] << " ";
    // }
    // cout << endl;




    // t = t.transpose(1, 0);

    // cout << t;
//
//    vector<int> indexing_vector = vector<int>({-1, 1});
//    t = t.indexing(indexing_vector);
//    cout << t;

    vector<int> shape = {3, 3, 3};
    int data1[27] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15, 16, 17, 18,
                     19, 20, 21, 22, 23, 24, 25, 26, 27};
    ts::Tensor<int> t1 = ts::Tensor<int>(data1, shape);
    int data2[27] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15, 16, 17, 18,
                     19, 20, 21, 22, 23, 24, 25, 26, 27};
    ts::Tensor<int> t2 = ts::Tensor<int>(data2, shape);
    ts::Tensor<bool> compare = t1.gt(t2);
    cout << compare << endl;

//    cout << t << endl;
//    ts::Tensor sum = ts::sum(t, 0);
//    cout << sum << endl;

//    ts::Tensor mean = ts::mean(t, 2);
//    cout << mean << endl;
    ts::Tensor min = ts::min(t1, 1);
    cout << min << endl;
//    ts::Tensor max = ts::max(t, 0);
//    cout << max << endl;

    return 0;
}