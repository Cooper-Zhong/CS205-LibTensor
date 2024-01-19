#include "tensor.h"
#include "test.h"

using namespace std;

void test_cu_add(){
    // test tile
    vector<int> shape = {2, 2, 2};
    ts::Tensor<double> a1 = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> a2 = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> a3 = ts::Tensor<double>(shape);

    cout << a1 << endl;
    cout << a2 << endl;

    a3 = a1.cu_add(a2);

    cout << a3 << endl;
    a1.gpu_free();
    a2.gpu_free();
}

int main(){
    test_cu_add();
}