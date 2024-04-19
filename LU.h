#ifndef LU_H
#define LU_H

#include <iostream>
#include <vector>
#include "matrix.h"
using namespace std;
vector<double> LU_decomp(Matrix A, vector<double> F);
pair<Matrix, Matrix> get_L_and_U(Matrix A);
vector<double> forwardLU(Matrix L, vector<double> F);
vector<double> backwardLU(Matrix U, vector<double> y);
#endif 