#include "tensor.h"

using namespace std;

void general_test_acceleration(){
    vector<int> shape = {3,4};
    double *data = new double[12]{0,0,0,0,4,5,6,7,11,11,11,11};
    ts::Tensor<double> t1 = ts::arange<double>(0,12).reshape(shape);
    ts::Tensor<double> t2 = ts::Tensor<double>(data,shape);

    t1.gpu();
    t2.gpu();

    ts::Tensor<double> t_val = ts::Tensor<double>(shape);
    ts::Tensor<bool> t_bool = ts::Tensor<bool>(shape);
    cout << "Beginning testing for acceleration in the sequence:\ndefault\nomp\ncuda" << endl;
    cout << "t1: \n" << t1;
    cout << "t2: \n" << t2;
    cout << "--------------- Addition Test ---------------" << endl;
    t_val = t1 + t2;
    cout << t_val;
    t_val = t1.omp_add(t2);
    cout << t_val;
    t_val = t1.cu_add(t2);
    cout << t_val;

    cout << "--------------- Subtraction Test ---------------" << endl;
    t_val = t1 - t2;
    cout << t_val;
    t_val = t1.omp_sub(t2);
    cout << t_val;
    t_val = t1.cu_sub(t2);
    cout << t_val;

    cout << "--------------- Multiplication Test ---------------" << endl;
    t_val = t1 * t2;
    cout << t_val;
    t_val = t1.omp_mul(t2);
    cout << t_val;
    t_val = t1.cu_mul(t2);
    cout << t_val;

    cout << "--------------- Division Test ---------------" << endl;
    t_val = t1 / t2;
    cout << t_val;
    t_val = t1.omp_div(t2);
    cout << t_val;
    t_val = t1.cu_div(t2);
    cout << t_val;

    cout << "--------------- Logarithm Test ---------------" << endl;
    t_val = t1.log();
    cout << t_val;
    t_val = t1.omp_log();
    cout << t_val;
    t_val = t1.cu_log();
    cout << t_val;



    cout << "--------------- Equality Test ---------------" << endl;
    t_bool = t1.eq(t2);
    cout << t_bool;
    t_bool = t1.omp_eq(t2);
    cout << t_bool;
    t_bool = t1.cu_eq(t2);
    cout << t_bool;

    cout << "--------------- Inequality Test ---------------" << endl;
    t_bool = t1.ne(t2);
    cout << t_bool;
    t_bool = t1.omp_ne(t2);
    cout << t_bool;
    t_bool = t1.cu_ne(t2);
    cout << t_bool;

    cout << "--------------- Greater Than Test ---------------" << endl;
    t_bool = t1.gt(t2);
    cout << t_bool;
    t_bool = t1.omp_gt(t2);
    cout << t_bool;
    t_bool = t1.cu_gt(t2);
    cout << t_bool;

    cout << "--------------- Greater Equal Test ---------------" << endl;
    t_bool = t1.ge(t2);
    cout << t_bool;
    t_bool = t1.omp_ge(t2);
    cout << t_bool;
    t_bool = t1.cu_ge(t2);
    cout << t_bool;

    cout << "--------------- Less Than Test ---------------" << endl;
    t_bool = t1.lt(t2);
    cout << t_bool;
    t_bool = t1.omp_lt(t2);
    cout << t_bool;
    t_bool = t1.cu_lt(t2);
    cout << t_bool;

    cout << "--------------- Less Equal Test ---------------" << endl;
    t_bool = t1.le(t2);
    cout << t_bool;
    t_bool = t1.omp_le(t2);
    cout << t_bool;
    t_bool = t1.cu_le(t2);
    cout << t_bool;





    



}




int main(){
    general_test_acceleration();
}