#include <vector>
#include <iostream>

#include "tensor.h"

using namespace std;

int main(){
    vector<int> size = {3,4,2};
    vector<int> size_1 = {4,4,4};
    double data[6]={0.1, 1.2, 2.2, 3.1, 4.9, 5.2};
    ts::Tensor<double> t = ts::Tensor<double>::eye_tensor(size_1);
    
    cout << t.get_type() << endl;
    cout << t.get_data() << endl;
    auto t_data=t.get_data();
    for (int i = 0; i < t.get_data_length(); i++)
    {
        cout << t_data[i] << " ";
    }
    cout << endl;
    
    cout << t.get_offset() << endl;
    cout << t.get_ndim() << endl;
    cout << t.get_data_length() << endl;

    auto t_shape = t.get_shape();
    auto t_stride = t.get_stride();

    for (int i = 0; i < t_shape.size(); i++)
    {
        cout << t_shape[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < t_stride.size(); i++)
    {
        cout << t_stride[i] << " ";
    }
    cout << endl;

    cout << t;
    


    return 0;
}