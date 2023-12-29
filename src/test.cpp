#include <vector>
#include "tensor.h"
#include <iostream>

using namespace std;

int main(){
    vector<int> size = {3,2};
    ts::Tensor<double> t1 = ts::Tensor<double>(size);
    return 0;
}