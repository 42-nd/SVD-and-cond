#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>

using namespace std;

class Matrix {
public:
    int dim;
    vector<vector<double>> data;
    Matrix(int dim);
    void print() const;
    Matrix dotProduct(const Matrix& other) const;
    vector<double> vecProduct(const vector<double>& vec) const;
    Matrix transpose() const;
    void Column_Transposition(int i, int j);
};

#endif 
