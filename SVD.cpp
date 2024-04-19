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
    //U.print();
    //cout << "------------------" << endl;
    //V.print();
    //cout << "------------------" << endl;
   /* S.print();
    cout << "|||||||||||||||||||||||||||||||" << endl;*/
    return make_pair(x,S.data[0][0]/S.data[rank-1][rank-1]);
}

void Start_SVD(Matrix& U, Matrix& Sigma, Matrix& V, Matrix A) {
    int n = A.dim;

    // инициализация матриц для SVD
    for (int i = 0; i < n; i++) {
        U.data[i][i] = 1.0;
        for (int j = 0; j < n; j++) {
            Sigma.data[i][j] = A.data[i][j];
        }
    }
    for (int i = 0; i < n; i++) {
        V.data[i][i] = 1.0;
    }

    // **************** этап I: бидиагонализация *************************
    for (int i = 0; i < n-1; i++) {
        Column_Transformation(Sigma, U, i, i);
        Row_Transformation(Sigma, V, i, i + 1);
    }

    // **************** этап II: преследование ************
    // ********* приведение к диагональному виду **********

    // для хранение изменяющихся элементов верхней диагонали
    vector<double> Up(n-1);
    vector<double> Down(n-1);
    // число неизменившихся элементов над главной диагональю
    int CountUpElements;

    // процедура преследования
    do {
        CountUpElements = 0;

        // обнуление верхней диагонали
        for (int i = 0; i < n-1; i++) {
            if (fabs(Up[i] - Sigma.data[i][i + 1]) > CONST_EPS) {
                Up[i] = Sigma.data[i][i + 1];
                Delete_Elem_Up_Triangle(Sigma, V, i, i + 1);
            }
            else
                CountUpElements++;
        }

        // обнуление нижней диагонали
        for (int i = 0; i < n-1; i++) {
            if (fabs(Down[i] - Sigma.data[i + 1][i]) > CONST_EPS) {
                Down[i] = Sigma.data[i + 1][i];
                Delete_Elem_Down_Triangle(Sigma, U, i + 1, i);
            }
        }
    } while (CountUpElements != n-1);

    // убираем отрицательные сингулярные числа
    Check_Singular_Values(Sigma,U);
    // сортируем по возрастанию сингулярные числа
    Sort_Singular_Values(Sigma,U,V);
}

void Column_Transformation(Matrix& A, Matrix& U, int i, int j) {
    // вектор отражения
    int dim = A.dim;
    vector<double> p(dim);

    // вспомогательные переменные
    double s, beta, mu;

    // находим квадрат нормы столбца для обнуления
    s = 0;
    for (int I = j; I < dim; I++) s += pow(A.data[I][i], 2);

    // если ненулевые элементы под диагональю есть:
    // квадрат нормы вектора для обнуления не совпадает с квадратом зануляемого элемента
    if (sqrt(fabs(s - A.data[j][i] * A.data[j][i])) > CONST_EPS) {
        // выбор знака слагаемого beta = sign(-x1)
        if (A.data[j][i] < 0) beta = sqrt(s);
        else beta = -sqrt(s);

        // вычисляем множитель в м.Хаусхолдера mu = 2 / ||p||^2
        mu = 1.0 / beta / (beta - A.data[j][i]);

        // формируем вектор p
        for (int I = 0; I < dim; I++) { p[I] = 0; if (I >= j) p[I] = A.data[I][i]; }

        // изменяем элемент, с которого начнётся обнуление
        p[j] -= beta;

        // вычисляем новые компоненты матрицы A = ... * U2 * U1 * A
        for (int m = 0; m < dim; m++) {
            // произведение S = St * p
            s = 0;
            for (int n = j; n < dim; n++) { s += A.data[n][m] * p[n]; }
            s *= mu;
            // S = S - 2 * p * (St * p)^t / ||p||^2
            for (int n = j; n < dim; n++) { A.data[n][m] -= s * p[n]; }
        }

        // вычисляем новые компоненты матрицы U = ... * H2 * H1 * U
        for (int m = 0; m < dim; m++) {
            // произведение S = Ut * p
            s = 0;
            for (int n = j; n < dim; n++) { s += U.data[m][n] * p[n]; }
            s *= mu;
            // U = U - 2 * p * (Ut * p)^t / ||p||^2
            for (int n = j; n < dim; n++) { U.data[m][n] -= s * p[n]; }
        }
    }
}

void Row_Transformation(Matrix& A, Matrix& V, int i, int j) {
    // вектор отражения
    int dim = A.dim;
    vector<double> p(dim);

    // вспомогательные переменные
    double s, beta, mu;

    // находим квадрат нормы строки для обнуления
    s = 0;
    for (int I = j; I < dim; I++) s += pow(A.data[i][I], 2);

    // если ненулевые элементы под диагональю есть:
    // квадрат нормы вектора для обнуления не совпадает с квадратом зануляемого элемента
    if (sqrt(fabs(s - A.data[i][j] * A.data[i][j])) > CONST_EPS) {
        // выбор знака слагаемого beta = sign(-x1)
        if (A.data[i][j] < 0) beta = sqrt(s);
        else beta = -sqrt(s);

        // вычисляем множитель в м.Хаусхолдера mu = 2 / ||p||^2
        mu = 1.0 / beta / (beta - A.data[i][j]);

        // формируем вектор p
        for (int I = 0; I < dim; I++) { p[I] = 0; if (I >= j) p[I] = A.data[i][I]; }

        // изменяем диагональный элемент
        p[j] -= beta;

        // вычисляем новые компоненты матрицы A = A * H1 * H2 ...
        for (int m = 0; m < dim; m++) {
            // произведение A * p
            s = 0;
            for (int n = j; n < dim; n++) { s += A.data[m][n] * p[n]; }
            s *= mu;
            // A = A - p * (A * p)^t
            for (int n = j; n < dim; n++) { A.data[m][n] -= s * p[n]; }
        }

        // вычисляем новые компоненты матрицы V = V * H1 * H2 * ...
        for (int m = 0; m < dim; m++) {
            // произведение V * p
            s = 0;
            for (int n = j; n < dim; n++) { s += V.data[m][n] * p[n]; }
            s *= mu;
            // V = V - p * (V * p)^t
            for (int n = j; n < dim; n++) { V.data[m][n] -= s * p[n]; }
        }
    }
}

void Delete_Elem_Down_Triangle(Matrix& A, Matrix& U, int I, int J) {
    double help1, help2;

    // косинус, синус
    double c = 0, s = 0;

    // если элемент не нулевой, то требуется поворот вектора
    if (fabs(A.data[I][J]) > CONST_EPS) {
        help1 = sqrt(pow(A.data[I][J], 2) + pow(A.data[J][J], 2));
        c = A.data[J][J] / help1;
        s = A.data[I][J] / help1;

        // A_new = Gt * A
        for (int k = 0; k < A.dim; k++) {
            help1 = c * A.data[J][k] + s * A.data[I][k];
            help2 = c * A.data[I][k] - s * A.data[J][k];
            A.data[J][k] = help1;
            A.data[I][k] = help2;
        }
        // умножаем матрицу U на матрицу преобразования G справа: D = Qt * A * Q => Qt транспонируется для матрицы U 
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

    // косинус, синус
    double c = 0, s = 0;

    // если элемент не нулевой, то требуется поворот вектора
    if (fabs(A.data[I][J]) > CONST_EPS) {
        help1 = sqrt(pow(A.data[I][J], 2) + pow(A.data[I][I], 2));
        c = A.data[I][I] / help1;
        s = -A.data[I][J] / help1;

        // A_new = A * Gt
        for (int k = 0; k < A.dim; k++) {
            help1 = c * A.data[k][I] - s * A.data[k][J];
            help2 = c * A.data[k][J] + s * A.data[k][I];
            A.data[k][I] = help1;
            A.data[k][J] = help2;
        }
        // умножаем матрицу V на матрицу преобразования Gt справа 
        for (int k = 0; k < V.dim; k++) {
            help1 = c * V.data[k][I] - s * V.data[k][J];
            help2 = c * V.data[k][J] + s * V.data[k][I];
            V.data[k][I] = help1;
            V.data[k][J] = help2;
        }
    }
}

void Check_Singular_Values(Matrix& Sigma, Matrix& U) {
    // наименьшее измерение
    int Min_Size = Sigma.dim;

    // проверка сингулярных чисел на положительность
    for (int i = 0; i < Min_Size; i++) {
        if (Sigma.data[i][i] < 0) {
            Sigma.data[i][i] = -Sigma.data[i][i];

            for (int j = 0; j < U.dim; j++)
                U.data[j][i] = -U.data[j][i];
        }
    }
}

void Sort_Singular_Values(Matrix& Sigma, Matrix& U, Matrix& V) {
    // наименьшее измерение
    int Min_Size = Sigma.dim;

    // сортировка сингулярных чисел
    for (int I = 0; I < Min_Size; I++) {
        double Max_Elem = Sigma.data[I][I];
        int Index = I;
        for (int i = I + 1; i < Min_Size; i++) {
            if (Sigma.data[i][i] > Max_Elem) {
                Max_Elem = Sigma.data[i][i];
                Index = i;
            }
        }
        // найден наибольший элемент
        if (I != Index) {
            // Обмен значений наибольшего элемента и текущего элемента
            double temp = Sigma.data[Index][Index];
            Sigma.data[Index][Index] = Sigma.data[I][I];
            Sigma.data[I][I] = temp;

            // Транспонирование столбцов матриц U и V
            U.Column_Transposition(I, Index);
            V.Column_Transposition(I, Index);
        }
    }
}
