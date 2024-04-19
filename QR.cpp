#include "QR.h" 

using namespace std;
vector<double> QR_decomp(Matrix A, vector<double> F) {
    pair<Matrix, Matrix> QR = get_Q_and_R(A);
    vector<double> y = QR.first.transpose().vecProduct(F);
    vector<double> x = backwardQR(QR.second, y, y);
    return x;
}
pair<Matrix, Matrix> get_Q_and_R(Matrix A) {
    int N = A.dim;
    Matrix Q(N), R(N);
    for (int i = 0; i < N; i++) {
        Q.data[i][i] = 1.0;
    }
    vector<double> p(N);
    R = A;

    for (int i = 0; i < N - 1; i++) {
        double s = 0;
        for (int I = i; I < N; I++)
            s += R.data[I][i] * R.data[I][i];

        if (sqrt(abs(s - R.data[i][i] * R.data[i][i])) > 1e-15) {
            double beta = (R.data[i][i] < 0) ? sqrt(s) : -sqrt(s);
            double mu = 1.0 / beta / (beta - R.data[i][i]);

            for (int I = 0; I < N; I++) {
                p[I] = (I >= i) ? R.data[I][i] : 0;
            }
            p[i] -= beta;

            for (int m = i; m < N; m++) {
                double s = 0;
                for (int n = i; n < N; n++) {
                    s += R.data[n][m] * p[n];
                }
                s *= mu;
                for (int n = i; n < N; n++) {
                    R.data[n][m] -= s * p[n];
                }
            }

            for (int m = 0; m < N; m++) {
                double s = 0;
                for (int n = i; n < N; n++) {
                    s += Q.data[m][n] * p[n];
                }
                s *= mu;
                for (int n = i; n < N; n++) {
                    Q.data[m][n] -= s * p[n];
                }
            }
        }
    }

    return make_pair(Q, R);
}

vector<double> backwardQR(Matrix A, vector<double> F, vector<double> y)  {
    int dim = A.dim;
    y = F;
    for (int i = dim - 1; i >= 0; --i) {
        for (int j = i + 1; j < dim; ++j) {
            y[i] -= A.data[i][j] * y[j];
        }
        y[i] /= A.data[i][i];
    }
    return y;
}