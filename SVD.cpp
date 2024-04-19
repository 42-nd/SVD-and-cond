#include "SVD.h" 
const double CONST_EPS = 1e-30;
pair<vector<double>, double> SVD_solver_and_cond(Matrix A, vector<double> F) {
    Matrix U(A.dim);
    Matrix V(A.dim);
    Matrix S(A.dim);
    Start_SVD(U, S, V, A);
    vector<double> UtF(A.dim);
    UtF = U.transpose().vecProduct(F);
    for (int i = 0; i < A.dim; i++) 
        UtF[i] /= S.data[i][i];
    vector<double> x = V.vecProduct(UtF);
    int rank = S.dim;
    return make_pair(x,S.data[0][0]/S.data[rank-1][rank-1]);
}

void Start_SVD(Matrix& U, Matrix& Sigma, Matrix& V, Matrix A) {
    int n = A.dim;

    for (int i = 0; i < n; i++) {
        U.data[i][i] = 1.0;
        for (int j = 0; j < n; j++) {
            Sigma.data[i][j] = A.data[i][j];
        }
    }
    for (int i = 0; i < n; i++) {
        V.data[i][i] = 1.0;
    }


    for (int i = 0; i < n-1; i++) {
        Column_Transformation(Sigma, U, i, i);
        Row_Transformation(Sigma, V, i, i + 1);
    }


    vector<double> Up(n-1);
    vector<double> Down(n-1);
    int CountUpElements;

    do {
        CountUpElements = 0;

        for (int i = 0; i < n-1; i++) {
            if (fabs(Up[i] - Sigma.data[i][i + 1]) > CONST_EPS) {
                Up[i] = Sigma.data[i][i + 1];
                Delete_Elem_Up_Triangle(Sigma, V, i, i + 1);
            }
            else
                CountUpElements++;
        }

        for (int i = 0; i < n-1; i++) {
            if (fabs(Down[i] - Sigma.data[i + 1][i]) > CONST_EPS) {
                Down[i] = Sigma.data[i + 1][i];
                Delete_Elem_Down_Triangle(Sigma, U, i + 1, i);
            }
        }
    } while (CountUpElements != n-1);

    Check_Singular_Values(Sigma,U);
    Sort_Singular_Values(Sigma,U,V);
}

void Column_Transformation(Matrix& A, Matrix& U, int i, int j) {
    int dim = A.dim;
    vector<double> p(dim);

    double s, beta, mu;

    s = 0;
    for (int I = j; I < dim; I++) s += pow(A.data[I][i], 2);

    if (sqrt(fabs(s - A.data[j][i] * A.data[j][i])) > CONST_EPS) {
        if (A.data[j][i] < 0) beta = sqrt(s);
        else beta = -sqrt(s);

        mu = 1.0 / beta / (beta - A.data[j][i]);

        for (int I = 0; I < dim; I++) { p[I] = 0; if (I >= j) p[I] = A.data[I][i]; }

        p[j] -= beta;

        for (int m = 0; m < dim; m++) {
            s = 0;
            for (int n = j; n < dim; n++) { s += A.data[n][m] * p[n]; }
            s *= mu;
            for (int n = j; n < dim; n++) { A.data[n][m] -= s * p[n]; }
        }

        for (int m = 0; m < dim; m++) {
            s = 0;
            for (int n = j; n < dim; n++) { s += U.data[m][n] * p[n]; }
            s *= mu;
            for (int n = j; n < dim; n++) { U.data[m][n] -= s * p[n]; }
        }
    }
}

void Row_Transformation(Matrix& A, Matrix& V, int i, int j) {
    int dim = A.dim;
    vector<double> p(dim);
    double s, beta, mu;
    s = 0;
    for (int I = j; I < dim; I++) s += pow(A.data[i][I], 2);
    if (sqrt(fabs(s - A.data[i][j] * A.data[i][j])) > CONST_EPS) {
        if (A.data[i][j] < 0) beta = sqrt(s);
        else beta = -sqrt(s);

        mu = 1.0 / beta / (beta - A.data[i][j]);

        for (int I = 0; I < dim; I++) { p[I] = 0; if (I >= j) p[I] = A.data[i][I]; }

        p[j] -= beta;

        for (int m = 0; m < dim; m++) {
            s = 0;
            for (int n = j; n < dim; n++) { s += A.data[m][n] * p[n]; }
            s *= mu;
            for (int n = j; n < dim; n++) { A.data[m][n] -= s * p[n]; }
        }

        for (int m = 0; m < dim; m++) {
            s = 0;
            for (int n = j; n < dim; n++) { s += V.data[m][n] * p[n]; }
            s *= mu;
            for (int n = j; n < dim; n++) { V.data[m][n] -= s * p[n]; }
        }
    }
}

void Delete_Elem_Down_Triangle(Matrix& A, Matrix& U, int I, int J) {
    double help1, help2;

    double c = 0, s = 0;

    if (fabs(A.data[I][J]) > CONST_EPS) {
        help1 = sqrt(pow(A.data[I][J], 2) + pow(A.data[J][J], 2));
        c = A.data[J][J] / help1;
        s = A.data[I][J] / help1;

        for (int k = 0; k < A.dim; k++) {
            help1 = c * A.data[J][k] + s * A.data[I][k];
            help2 = c * A.data[I][k] - s * A.data[J][k];
            A.data[J][k] = help1;
            A.data[I][k] = help2;
        }
        for (int k = 0; k < U.dim; k++) {
            help1 = c * U.data[k][J] + s * U.data[k][I];
            help2 = c * U.data[k][I] - s * U.data[k][J];
            U.data[k][J] = help1;
            U.data[k][I] = help2;
        }
    }
    A.data[I][J] = 0;
}

void Delete_Elem_Up_Triangle(Matrix& A, Matrix& V, int I, int J) {
    double help1, help2;

    double c = 0, s = 0;

    if (fabs(A.data[I][J]) > CONST_EPS) {
        help1 = sqrt(pow(A.data[I][J], 2) + pow(A.data[I][I], 2));
        c = A.data[I][I] / help1;
        s = -A.data[I][J] / help1;

        for (int k = 0; k < A.dim; k++) {
            help1 = c * A.data[k][I] - s * A.data[k][J];
            help2 = c * A.data[k][J] + s * A.data[k][I];
            A.data[k][I] = help1;
            A.data[k][J] = help2;
        }
        for (int k = 0; k < V.dim; k++) {
            help1 = c * V.data[k][I] - s * V.data[k][J];
            help2 = c * V.data[k][J] + s * V.data[k][I];
            V.data[k][I] = help1;
            V.data[k][J] = help2;
        }
    }
}

void Check_Singular_Values(Matrix& Sigma, Matrix& U) {
    int Min_Size = Sigma.dim;

    for (int i = 0; i < Min_Size; i++) {
        if (Sigma.data[i][i] < 0) {
            Sigma.data[i][i] = -Sigma.data[i][i];

            for (int j = 0; j < U.dim; j++)
                U.data[j][i] = -U.data[j][i];
        }
    }
}

void Sort_Singular_Values(Matrix& Sigma, Matrix& U, Matrix& V) {
    int Min_Size = Sigma.dim;

    for (int I = 0; I < Min_Size; I++) {
        double Max_Elem = Sigma.data[I][I];
        int Index = I;
        for (int i = I + 1; i < Min_Size; i++) {
            if (Sigma.data[i][i] > Max_Elem) {
                Max_Elem = Sigma.data[i][i];
                Index = i;
            }
        }
        if (I != Index) {
            double temp = Sigma.data[Index][Index];
            Sigma.data[Index][Index] = Sigma.data[I][I];
            Sigma.data[I][I] = temp;
            U.Column_Transposition(I, Index);
            V.Column_Transposition(I, Index);
        }
    }
}
