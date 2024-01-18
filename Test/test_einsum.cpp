#include "tensor.h"
#include <vector>
#include <iostream>

using namespace std;

int main()
{

    cout << "Test 1: ================================" << endl;
    ts::Tensor<int> t1 = ts::arange<int>(0, 60).reshape({3, 4, 5});
//    cout << "t1:\n"
//         << t1 << endl;
    ts::Tensor<int> t2 = ts::arange<int>(0, 60).reshape({3, 5, 4});
//    cout << "t2:\n"
//         << t2 << endl;
    ts::Tensor<int> t3 = ts::einsum<int>("ijk,ikl->ij", {t1, t2});
    cout << "t3:" << t3 << endl;

    cout << "Test 2: ================================" << endl;
    ts::Tensor<int> t7 = ts::arange<int>(0, 9).reshape({3, 3});
//    std::cout << "t7:\n"
//              << t7 << std::endl;
    ts::Tensor<int> t8 = ts::einsum<int>("ii->i", {t7});
    std::cout << "t8:\n"
              << t8 << std::endl;

    cout << "Test 3: ================================" << endl;
    ts::Tensor<int> t9 = ts::arange<int>(0, 6).reshape({2, 3});
//    std::cout << "t9:\n"
//              << t9 << std::endl;
    ts::Tensor<int> t10 = ts::einsum<int>("ij->", {t9});
    std::cout << "t10:\n"
              << t10 << std::endl;

    cout << "Test 4: ================================" << endl;
    ts::Tensor<int> t11 = ts::arange<int>(0, 6).reshape({2, 3});
//    std::cout << "t11:\n"
//              << t11 << std::endl;
    ts::Tensor<int> t12 = ts::einsum<int>("ij->j", {t11});
    std::cout << "t12:\n"
              << t12 << std::endl;

    cout << "Test 5: ================================" << endl;
    ts::Tensor<int> t13 = ts::arange<int>(0, 6).reshape({2, 3});
//    std::cout << "t13:\n"
//              << t13 << std::endl;
    ts::Tensor<int> t14 = ts::arange<int>(0, 3);
//    std::cout << "t14:\n"
//              << t14 << std::endl;
    ts::Tensor<int> t15 = ts::einsum<int>("ij,j->i", {t13, t14});
    std::cout << "t15:\n"
              << t15 << std::endl;

    cout << "Test 6: ================================" << endl;
    ts::Tensor<int> t16 = ts::arange<int>(0, 6).reshape({2, 3});
//    std::cout << "t16:\n"
//              << t16 << std::endl;
    ts::Tensor<int> t17 = ts::arange<int>(0,15).reshape({3,5});
//    std::cout << "t17:\n"
//              << t17 << std::endl;
    ts::Tensor<int> t18 = ts::einsum<int>("ij,jk->ik", {t16, t17});
    std::cout << "t18:\n"
              << t18 << std::endl;

    cout << "Test 7: ================================" << endl;
    ts::Tensor<int> t19 = ts::arange<int>(0, 6).reshape({2, 3});
//    std::cout << "t19:\n"
//              << t19 << std::endl;
    ts::Tensor<int> t20 = ts::arange<int>(6,12).reshape({2,3});
//    std::cout << "t20:\n"
//              << t20 << std::endl;
    ts::Tensor<int> t21 = ts::einsum<int>("ij,ij->", {t19, t20});
    std::cout << "t21:\n"
              << t21 << std::endl;

    cout << "Test 8: ================================" << endl;

    ts::Tensor<int> t22 = ts::arange<int>(0, 3);
//    std::cout << "t22:\n"
//              << t22 << std::endl;
    ts::Tensor<int> t23 = ts::arange<int>(3,7);
//    std::cout << "t23:\n"
//              << t23 << std::endl;
    ts::Tensor<int> t24 = ts::einsum<int>("i,j->ij", {t22, t23});
    std::cout << "t24:\n"
              << t24 << std::endl;

    cout << "Test 9: ================================" << endl;
    ts::Tensor<int> t25 = ts::arange<int>(0, 30).reshape({2, 3, 5});
//    std::cout << "t25:\n"
//              << t25 << std::endl;
    ts::Tensor<int> t26 = ts::arange<int>(0, 40).reshape({2,5,4});
//    std::cout << "t26:\n"
//              << t26 << std::endl;
    ts::Tensor<int> t27 = ts::einsum<int>("ijk,ikl->ijl", {t25, t26});
    std::cout << "t27:\n"
              << t27 << std::endl;

    cout << "Test 10: ================================" << endl;
    ts::Tensor<int> t28 = ts::arange<int>(0, 6).reshape({2, 3});
//    std::cout << "t28:\n"
//              << t28 << std::endl;
    ts::Tensor<int> t29 = ts::arange<int>(0, 105).reshape({5,3,7});
//    std::cout << "t29:\n"
//              << t29 << std::endl;
    ts::Tensor<int> t30 = ts::arange<int>(0, 14).reshape({2,7});
//    std::cout << "t30:\n"
//              << t30 << std::endl;
    ts::Tensor<int> t31 = ts::einsum<int>("ik,jkl,il->ij", {t28, t29, t30});
    std::cout << "t31:\n"
              << t31 << std::endl;

    cout << "Test 11: ================================" << endl;
    ts::Tensor<int> t32 = ts::arange<int>(1,5).reshape({2,2});
//    std::cout << "t32:\n"
//              << t32 << std::endl;
    ts::Tensor<int> t33 = ts::arange<int>(5,9).reshape({2,2});
//    std::cout << "t33:\n"
//              << t33 << std::endl;
    ts::Tensor<int> t34 = ts::arange<int>(9,13).reshape({2,2});
//    std::cout << "t34:\n"
//              << t34 << std::endl;
    ts::Tensor<int> t35 = ts::einsum<int>("ij,ik,jk->", {t32, t33, t34});
    std::cout << "t35:\n"
              << t35 << std::endl;

    // cout << "Test 11: ================================" << endl;
    // ts::Tensor<int> t32 = ts::arange<int>(0, 210).reshape({2, 3, 5, 7});
    // ts::Tensor<int> t33 = ts::arange<int>(0, 36465).reshape({11,13,3,17,5});
    // ts::Tensor<int> t34 = ts::einsum<int>("pqrs,tuqvr->pstuv", {t32, t33});
    // std::cout << "t34:\n"
    //           << t34 << std::endl;
    



    return 0;
}