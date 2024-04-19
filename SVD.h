#ifndef SVD_H
#define SVD_H

#include "matrix.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
pair<vector<double>,double> SVD_solver_and_cond(Matrix A, vector<double> F);
void Start_SVD(Matrix& U, Matrix& Sigma, Matrix& V, Matrix A);
void Column_Transformation(Matrix& A, Matrix& U, int i, int j);
void Row_Transformation(Matrix& A, Matrix& V, int i, int j);
void Delete_Elem_Down_Triangle(Matrix& A, Matrix& U, int I, int J);
void Delete_Elem_Up_Triangle(Matrix& A, Matrix& V, int I, int J);
void Check_Singular_Values(Matrix& Sigma, Matrix& U);
void Sort_Singular_Values(Matrix& Sigma, Matrix& U, Matrix& V);
#endif
