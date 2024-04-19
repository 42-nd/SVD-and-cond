#include "LU.h"

vector<double> LU_decomp(Matrix A, vector<double> F) {
	pair<Matrix, Matrix> LU = get_L_and_U(A);
	vector<double> y = forwardLU(LU.first, F);
	vector<double> x = backwardLU(LU.second, y);
	return x;
}
pair<Matrix, Matrix> get_L_and_U(Matrix A) {
    int dim = A.dim;
    Matrix L(dim), U(dim);
    U = A;
    for (int i = 0; i < dim; ++i) {
        for (int k = i; k < dim; ++k) {
            double sum = 0;
            for (int j = 0; j < i; ++j)
                sum += (L.data[i][j] * U.data[j][k]);
            U.data[i][k] = A.data[i][k] - sum;
        }

        for (int k = i; k < dim; ++k) {
            if (i == k)
                L.data[i][i] = 1; 
            else {
                double sum = 0;
                for (int j = 0; j < i; ++j)
                    sum += (L.data[k][j] * U.data[j][i]);
                L.data[k][i] = (A.data[k][i] - sum) / U.data[i][i];
            }
        }
    }
	return make_pair(L, U);
}

vector<double> forwardLU(Matrix L, vector<double> F) {
	int n = L.dim;
	vector<double> y(n, 0);
	for (int i = 0; i < n; ++i) {
		y[i] = F[i];
		for (int j = 0; j < i; ++j) {
			y[i] -= L.data[i][j] * y[j];
		}
		y[i] /= L.data[i][i];
	}
	return y;
}

vector<double> backwardLU(Matrix U, vector<double> y) {
	int n = U.dim;
	vector<double> x(n, 0);
	for (int i = n - 1; i >= 0; --i) {
		x[i] = y[i];
		for (int j = i + 1; j < n; ++j) {
			x[i] -= U.data[i][j] * x[j];
		}
		x[i] /= U.data[i][i];
	}
	return x;
}