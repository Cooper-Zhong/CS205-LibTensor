#include "tensor.h"
#include "test.h"

using namespace std;

void test_cu_add(){
    // test tile
    vector<int> shape1 = {2, 2};
    vector<int> shape2 = {2, 3};
    vector<int> shape3 = {2, 3};
    ts::Tensor<double> a1 = ts::Tensor<double>::rand_tensor(shape1);
    ts::Tensor<double> a2 = ts::Tensor<double>::rand_tensor(shape2);
    // ts::Tensor<double> a3 = ts::Tensor<double>(shape3);

    cout << a1 << endl;
    cout << a2 << endl;

    auto a3 = a1.cu_ein(a2);
    
    auto a4 = ts::einsum<double>("ij,jk->ik",{a1,a2});
    cout << a3 << endl;
    cout << a4 << endl;
    a1.gpu_free();
    a2.gpu_free();
}

int main(){
    test_cu_add();
}