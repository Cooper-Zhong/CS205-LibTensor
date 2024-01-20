#include "tensor.h"
#include "test.h"

using namespace std;

void test_serialize(){
    // test tile
    vector<int> shape = {2, 2, 2, 2, 2};
    ts::Tensor<double> a1 = create_test_tensor(shape, false, true);
    a1.serialize("test_serialization.bin");
    ts::Tensor<double> a2 = ts::Tensor<double>::deserialize("test_serialization.bin");
    cout << a2 << endl;
}

int main(){
    test_serialize();
}