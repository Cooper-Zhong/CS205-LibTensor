#include "tensor.h"
#include <vector>
#include <iostream>

using namespace std;

int main()
{
    ts::Tensor<int> t1, t2, t3;
    cout << "Test 1: ii->i " << endl;
    t1 = ts::arange<int>(0, 9).reshape({3, 3});
    cout << ts::einsum<int>("ii->i", {t1}) << endl;
    cout << "==============================" << endl;

    cout << "Test 2: ij->ji " << endl;
    t1 = ts::arange<int>(0, 6).reshape({2, 3});
    cout << ts::einsum<int>("ij->ji", {t1}) << endl;
    cout << "==============================" << endl;

    cout << "Test 3: ij->" << endl;
    t1 = ts::arange<int>(0, 6).reshape({2, 3});
    cout << ts::einsum<int>("ij->", {t1}) << endl;
    cout << "==============================" << endl;

    cout << "Test 4: ij->j" << endl;
    t1 = ts::arange<int>(0, 6).reshape({2, 3});
    cout << ts::einsum<int>("ij->j", {t1}) << endl;
    cout << "==============================" << endl;

    cout << "Test 5: ik,k->i" << endl;
    t1 = ts::arange<int>(0, 6).reshape({2, 3});
    t2 = ts::arange<int>(0, 3);
    cout << ts::einsum<int>("ik,k->i", {t1, t2}) << endl;
    cout << "==============================" << endl;

    cout << "Test 6: ik,kj->ij" << endl;
    t1 = ts::arange<int>(0, 6).reshape({2, 3});
    t2 = ts::arange<int>(0, 15).reshape({3, 5});
    cout << ts::einsum<int>("ik,kj->ij", {t1, t2}) << endl;
    cout << "==============================" << endl;

    cout << "Test 7: i,i->" << endl;
    t1 = ts::arange<int>(0, 6);
    cout << ts::einsum<int>("i,i->", {t1, t1}) << endl;
    cout << "==============================" << endl;

    cout << "Test 8: ij,ij->" << endl;
    t1 = ts::arange<int>(0, 6).reshape({2, 3});
    t2 = ts::arange<int>(6, 12).reshape({2, 3});
    cout << ts::einsum<int>("ij,ij->", {t1, t2}) << endl;
    cout << "==============================" << endl;

    cout << "Test 9: i,j->ij" << endl;
    t1 = ts::arange<int>(1, 6);
    t2 = ts::arange<int>(6, 11);
    cout << ts::einsum<int>("i,j->ij", {t1, t2}) << endl;
    cout << "==============================" << endl;

    cout << "Test 10: ijk,ikl->ijl" << endl;
    t1 = ts::arange<int>(0, 30).reshape({2, 3, 5});
    t2 = ts::arange<int>(0, 40).reshape({2, 5, 4});
    cout << ts::einsum<int>("ijk,ikl->ijl", {t1, t2}) << endl;
    cout << "==============================" << endl;

    cout << "Test 11: pqrs,tuqvr->pstuv" << endl;
    t1 = ts::arange<int>(0, 120).reshape({2, 3, 4, 5});
    t2 = ts::arange<int>(0, 288).reshape({2, 4, 3, 3, 4});
    cout << ts::einsum<int>("pqrs,tuqvr->pstuv", {t1, t2}) << endl;
    cout << "==============================" << endl;

    cout << "Test 12: ik,jkl,il->ij" << endl;
    t1 = ts::arange<int>(0, 6).reshape({2, 3});
    t2 = ts::arange<int>(0, 105).reshape({5, 3, 7});
    t3 = ts::arange<int>(0, 14).reshape({2, 7});
    cout << ts::einsum<int>("ik,jkl,il->ij", {t1, t2, t3}) << endl;
    cout << "==============================" << endl;

    return 0;
}