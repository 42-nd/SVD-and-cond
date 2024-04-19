#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "matrix.h"
#include "LU.h"
#include "QR.h"
#include "SVD.h"
using namespace std;

double get_error(vector<double> X_true, vector<double> X_found) {
	double norm_x_true = 0;
	double norm_x_diff = 0;
	int n = X_true.size();
	for (int i = 0; i < n; i++) {
		norm_x_true += pow(X_true[i],2);
	}
	norm_x_true = sqrt(norm_x_true);
	for (int i = 0; i < n; i++) {
		norm_x_diff += pow(X_true[i] - X_found[i],2);
	}
	norm_x_diff = sqrt(norm_x_diff);
	return norm_x_diff/norm_x_true;
}
int main() {
	int dims[3] = { 5,10,20};
	int iterations = 1000;
	cout << setprecision(15);
	cout << fixed;
	for (auto dim : dims) {

		Matrix A(dim);
		vector<double> X_true(dim);
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				A.data[i][j] = 1/(1 + 0.6 * i + 2 * j);
			}
			X_true[i] = 1;
		}
		vector<double> F(dim);
		F = A.vecProduct(X_true);
		double avg_time_qr = 0;
		double avg_time_lu = 0;
		double avg_time_SVD = 0;
		double avg_err_qr = 0; 
		double avg_err_lu = 0;
		double avg_err_SVD = 0;
		double cond = 0;

		cout << "Matrix dimension: " << dim << endl;
		cout << "Number of iterations: " << iterations << endl;
		for (int i = 0; i < iterations; i++) {
			auto start_qr = chrono::high_resolution_clock::now();
			vector<double> X_found_QR = QR_decomp(A, F);
			auto end_qr = std::chrono::high_resolution_clock::now();
			avg_time_qr += chrono::duration<double>(end_qr - start_qr).count();
			avg_err_qr += get_error(X_true, X_found_QR);

			auto start_lu = std::chrono::high_resolution_clock::now();
			vector<double> X_found_LU = LU_decomp(A, F);
			auto end_lu = std::chrono::high_resolution_clock::now();
			avg_time_lu += std::chrono::duration<double>(end_lu - start_lu).count();
			avg_err_lu += get_error(X_true, X_found_LU);

			auto start_SVD = std::chrono::high_resolution_clock::now();
			pair<vector<double>,double> X_found_SVD_and_cond = SVD_solver_and_cond(A, F);
			auto end_SVD = std::chrono::high_resolution_clock::now();
			avg_time_SVD += std::chrono::duration<double>(end_SVD - start_SVD).count();
			avg_err_SVD += get_error(X_true, X_found_SVD_and_cond.first);
			cond = X_found_SVD_and_cond.second;
			//if (i % 10 == 0) {
			//	cout << "Pass " << i << " iterations" << endl;
			//}
		}


		cout << "LU decomposition:" << endl;
		cout << "Average time: " << avg_time_lu/iterations << " seconds" << endl;
		cout << "Average error: " << avg_err_lu /iterations << endl;

		cout << "QR decomposition:" << endl;
		cout << "Average time: " << avg_time_qr/iterations << " seconds" << endl;
		cout << "Average error: " << avg_err_qr / iterations << endl;

		cout << "SVD decomposition:" << endl;
		cout << "Average time: " << avg_time_SVD / iterations << " seconds" << endl;
		cout << "Average error: " << avg_err_SVD / iterations << endl;
		cout << "Cond: " << cond << endl;
		cout << "---------------------------------------------------------------" << endl;
	}

	return 0;
}