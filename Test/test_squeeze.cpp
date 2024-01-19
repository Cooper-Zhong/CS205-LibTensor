#include "tensor.h"
#include "test.h"

using namespace std;

void test_squeeze(){
    // test tile
    vector<int> shape = {2, 1, 2};
    ts::Tensor<double> a1 = ts::Tensor<double>::rand_tensor(shape);

    cout << a1;
    for (int i = 0; i < 4; i++)
    {
        cout << a1.get_data()[i] << " ";
    }
    cout << endl;
    ts::Tensor<double> a2({2, 2});
    a2 = a1.squeeze();
    cout << a2 << endl;
    for (int i = 0; i < 4; i++)
    {
        cout << a2.get_data()[i] << " ";
    }
}

int main(){
    test_squeeze();
}