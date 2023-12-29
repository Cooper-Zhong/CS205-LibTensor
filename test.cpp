#include <vector>
#include "tensor.h"

using namespace std;

int main(){
    vector<int> size = {3,2};
    ts::Tensor<double> t = ts::Tensor<double>(size);
    double data[6]={0.1, 1.2, 2.2, 3.1, 4.9, 5.2};



    return 0;
}