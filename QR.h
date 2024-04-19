#ifndef QR_H
#define QR_H

#include "matrix.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
vector<double> QR_decomp(Matrix A, vector<double> F);
pair<Matrix, Matrix> get_Q_and_R(Matrix A);
vector<double> backwardQR(Matrix A, vector<double> F, vector<double> y);
#endif