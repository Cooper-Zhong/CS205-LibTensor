#ifndef TEST_H_
#define TEST_H_
#include <vector>
#include <iostream>
#include "tensor.h"
#include <algorithm>

using namespace std;

ts::Tensor<double> create_test_tensor(vector<int> shape, bool output_progress = false, bool output_final = true)
{
    int padding = 1;

    int data_length = 1;

    vector<int> padded_shape(shape.size());
    for (int i = 0; i < shape.size(); i++)
    {
        padded_shape[i] = shape[i] + padding * 2;
        data_length *= padded_shape[i];
    }
    double data[data_length];
    for (int i = 0; i < data_length; i++)
        data[i] = i;
    ts::Tensor<double> t(data, padded_shape);

    if (output_progress)
    {
        cout << "Test Tensor Created: " << endl
             << t << endl;
    }
    // randomly permute the axes
    vector<int> axes(shape.size());
    iota(axes.begin(), axes.end(), 0);
    shuffle(axes.begin(), axes.end(), std::mt19937(std::random_device()()));

    if (output_progress)
    {
        cout << "Randomly permuted axes: " << endl;
        for (auto i : axes)
            cout << i << " ";
        cout << endl;
    }

    // randomly permute the tensor
    ts::Tensor<double> t2 = t.permute(axes);

    if (output_progress)
    {
        cout << "Randomly permuted Tensor: " << endl
             << t2 << endl;
    }

    // slice the tensor
    vector<vector<int>> slice_indices;
    for (int i = 0; i < shape.size(); i++)
    {
        int start = padding;
        int end = padding + shape[i];
        slice_indices.push_back({start, end});
    }

    if (output_progress)
    {
        cout << "Sliced indices: " << endl;
        for (auto i : slice_indices)
        {
            cout << "[";
            for (auto j : i)
                cout << j << " ";
            cout << "] ";
        }
    }

    ts::Tensor<double> t3 = t2.slicing(slice_indices);

    if (output_progress)
    {
        cout << "Sliced Tensor: " << endl
             << t3 << endl;
    }
    /*
        // randomly permute the axes again
        iota(axes.begin(), axes.end(), 0);
        random_shuffle(axes.begin(), axes.end());

        if (output_progress)
        {
            cout << "Randomly permuted axes again: " << endl;
            for (auto i : axes)
                cout << i << " ";
            cout << endl;
        }

        // randomly permute the tensor
        ts::Tensor<double> t4 = t3.permute(axes);

        if (output_progress)
        {
            cout << "Randomly permuted Tensor again: " << endl
                 << t4 << endl;
        } */

    if (output_final)
        cout << "Final Tensor: " << endl
             << t3 << endl;

    return t3;
}
#endif