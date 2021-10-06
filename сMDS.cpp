#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#include <cmath>

#include <stdlib.h>
#include <omp.h>

// процедурный стиль
// camelCase

// Автор старался называть функции так, чтобы было по названию понятно, что она делает.
// Если автору кажется название неисчерпывающим, есть дополнительные комментарии.
// Выполнил Сенин Александр, 417

size_t numThreads = 8;

double generateRandomDouble(double lowerBound, double upperBound)
{
    // srand(42);
    double val = (double)rand() / RAND_MAX;

    return lowerBound + val * (upperBound - lowerBound);
}

void generateRandomCSVMatrix(size_t m, size_t n, const std::string &path="rand_matrix.csv", double lowerBound=-10000, double upperBound=10000) {
    std::ofstream outfile(path);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double randomDouble = generateRandomDouble(lowerBound, upperBound);
            outfile << randomDouble;
            if (j != n - 1) {
                outfile << ',';
            }
        }
        outfile << std::endl;
    }
}

void saveCSVMatrix(const std::vector<std::vector<double>> &matr, const std::string &path="saved.csv") {
    std::ofstream outfile(path);

    for (size_t i = 0; i < matr.size(); i++) {
        for (size_t j = 0; j < matr[0].size(); j++) {
            outfile << matr[i][j];
            if (j != matr[0].size() - 1) {
                outfile << ',';
            }
        }
        outfile << std::endl;
    }
}

std::vector<double> generateRandomVector(size_t n, double lowerBound=-10000, double upperBound=10000) {
    std::vector<double> res(n, 0);

    for (size_t i = 0; i < n; i++) {
        res[i] = generateRandomDouble(lowerBound, upperBound);
    }

    return res;
}

std::vector<double> readCSVRow(const std::string &s, char delim, int maxInLineIdx=-1) {
    std::vector<double> res;
    std::stringstream ss(s);
    std::string token;
    size_t iterIdx = 0;

    while (getline(ss, token, delim)) {
        if (maxInLineIdx != -1) {
            if (iterIdx > maxInLineIdx) {
                break;
            }
        }
        res.push_back(atof(token.c_str()));
        iterIdx += 1;
    }

    return res;
}

std::vector<std::vector<double>> readCSVMatrix(std::istream &in, int maxInLineIdx=-1) {
    std::vector<std::vector<double>> matrix;
    std::string row;

    while (!in.eof()) {
        std::getline(in, row);
        if (in.bad() || in.fail()) {
            break;
        }
        std::vector<double> values = readCSVRow(row, ',', maxInLineIdx);
        matrix.push_back(values);
    }

    return matrix;
}

std::vector<std::vector<double>> readMatrix(const std::string &path, int maxInLineIdx=-1) {
    std::ifstream file;
    file.open(path, std::ios::out);

    return readCSVMatrix(file, maxInLineIdx);
}

template<class T>
void printMatrix(const std::vector<std::vector<T>> &matrix) {
    for (std::vector<T> row: matrix) {
        for (T val: row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

template<class T>
void printVector(std::vector<T> &vect) {
    for (T val: vect) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

std::vector<double> matrixVectorMultiplyParallel(const std::vector<std::vector<double>> &matr, const std::vector<double> &vect) {
    // То самое Ar_k
    if (matr[0].size() != vect.size()) {
        throw "wrong matrix & vector shapes";
    }

    std::vector<double> res(matr.size(), 0);

    #pragma omp parallel num_threads(numThreads)
    {
        size_t i = 0, j = 0;
        double localSum = 0;
        #pragma omp for
        for (i = 0; i < matr.size(); i++) {
            localSum = 0;
            for (j = 0; j < vect.size(); j++) {
                localSum += matr[i][j] * vect[j];
            }
            res[i] = localSum;
        }
    }

    return res;
}

std::vector<std::vector<double>> matrixTransposedLambdaMultiply(
    const std::vector<std::vector<double>> &matr, const std::vector<std::vector<double>> &lambda
) {
    // Транспонируем матрицу и сразу же умножаем на скаляр
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < matr[0].size(); i++) {
        res.push_back(std::vector<double>(matr.size(), 0));
    }

    for (size_t i = 0; i < matr[0].size(); i++) {
        for (size_t j = 0; j < lambda.size(); j++) {
            res[i][j] = matr[j][i] * lambda[j][j];
        }
    }

    return res;
}

std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>> &matr, const std::vector<double> &vect) {
    if (matr[0].size() != vect.size()) {
        throw "wrong matrix & vector shapes";
    }

    size_t i = 0, j = 0;
    std::vector<double> res(matr.size(), 0);

    for (i = 0; i < matr.size(); i++) {
        for (j = 0; j < vect.size(); j++) {
            res[i] += matr[i][j] * vect[j];
        }
    }

    return res;
}

bool vectorEquals(const std::vector<double> &v1, const std::vector<double> &v2, double eps=0.001) {
    bool res = true;
    if (v1.size() != v2.size()) {
        throw "wrong vectors shapes";
    }

    for (size_t i = 0; i < v1.size(); i++) {
        if (abs(v1[i] - v2[i]) > eps) {
            res = false;
            break;
        }
    }

    return res;
}

bool matrixEquals(const std::vector<std::vector<double>> &v1, const std::vector<std::vector<double>> &v2, double eps=0.001) {
    bool res = true;
    if (v1.size() != v2.size()) {
        throw "wrong vectors shapes";
    }

    for (size_t i = 0; i < v1.size(); i++) {
        if (!vectorEquals(v1[i], v2[i])) {
            res = false;
            break;
        }
    }

    return res;
}

double innerProduct(const std::vector<double> &vect1, const std::vector<double> &vect2) {
    if (vect1.size() != vect2.size()) {
        throw "wrong vectors shapes";
    }

    double res = 0;
    for (size_t i = 0; i < vect1.size(); i++) {
        res += vect1[i] * vect2[i];
    }

    return res;
}

std::vector<std::vector<double>> matrixSelfTransposedMultiply(const std::vector<std::vector<double>> &matr) {
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < matr.size(); i++) {
        res.push_back(std::vector<double>(matr.size(), 0));
    }

    for (size_t i = 0; i < matr.size(); i++) {
        for (size_t j = i; j < matr.size(); j++) {
            res[i][j] = res[j][i] = innerProduct(matr[i], matr[j]);
        }
    }

    return res;
}

std::vector<std::vector<double>> matrixSelfTransposedMultiplyParallel(const std::vector<std::vector<double>> &matr) {
    // То самое XX^T
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < matr.size(); i++) {
        res.push_back(std::vector<double>(matr.size(), 0));
    }

    #pragma omp parallel num_threads(numThreads)
    {
        size_t i = 0, j = 0;
        #pragma omp for
        for (i = 0; i < matr.size(); i++) {
            for (j = i; j < matr.size(); j++) {
                res[i][j] = res[j][i] = innerProduct(matr[i], matr[j]);
            }
        }
    }

    return res;
}

std::vector<double> sumVector(const std::vector<double> &vect1, const std::vector<double> &vect2) {
    std::vector<double> res(vect1.size(), 0);

    for (size_t i = 0; i < vect1.size(); i++) {
        res[i] = vect1[i] + vect2[i];
    }

    return res;
}

std::vector<double> subVector(const std::vector<double> &vect1, const std::vector<double> &vect2) {
    std::vector<double> res(vect1.size(), 0);

    for (size_t i = 0; i < vect1.size(); i++) {
        res[i] = vect1[i] - vect2[i];
    }

    return res;
}

double norm(const std::vector<double> &vect) {
    double res = 0;

    for (size_t i = 0; i < vect.size(); ++i) {
        res += vect[i] * vect[i];
    }

    return sqrt(res);
}

std::vector<double> normalize(const std::vector<double> &vect) {
    double normalizer = norm(vect);

    std::vector<double> res(vect);
    for (size_t i = 0; i < vect.size(); i++) {
        res[i] = vect[i] / normalizer;
    }

    return res;
}

std::vector<double> powerIterationParallel(const std::vector<std::vector<double>> &matr, double &eigenValue, double eps=0.001, size_t maxIter=1000) {
    // Степенной метод
    if ((eps <= 0) || (maxIter == 0)) {
        throw "both stop rules are invalid";
    }

    std::vector<double> prevVector, currVector(matr[0].size(), 1);

    size_t currIter = 0;

    do {
        currIter += 1;
        prevVector = currVector;
        currVector = matrixVectorMultiplyParallel(matr, prevVector);
        currVector = normalize(currVector);
    } while ((currIter <= maxIter) && (norm(subVector(currVector, prevVector)) > eps));

    double eigenNom = innerProduct(currVector, matrixVectorMultiplyParallel(matr, currVector));
    double eigenDenom = innerProduct(currVector, currVector);

    eigenValue = eigenNom / eigenDenom;

    return currVector;
}

std::vector<std::vector<double>> vectorSelfTransposed(const std::vector<double> &vect) {
    // rr^T
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < vect.size(); i++) {
        res.push_back(std::vector<double>(vect.size(), 0));
    }

    for (size_t i = 0; i < vect.size(); i++) {
        for (size_t j = 0; j < vect.size(); j++) {
            res[i][j] = vect[i] * vect[j];
        }
    }

    return res;
}

std::vector<std::vector<double>> vectorSelfTransposedLambda(const std::vector<double> &vect, double lambda) {
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < vect.size(); i++) {
        res.push_back(std::vector<double>(vect.size(), 0));
    }

    for (size_t i = 0; i < vect.size(); i++) {
        for (size_t j = 0; j < vect.size(); j++) {
            res[i][j] = vect[i] * vect[j] * lambda;
        }
    }

    return res;
}

std::vector<std::vector<double>> matrixScalarMultiply(const std::vector<std::vector<double>> &matr, double mul) {
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < matr.size(); i++) {
        res.push_back(std::vector<double>(matr.size(), 0));
    }

    for (size_t i = 0; i < matr.size(); i++) {
        for (size_t j = 0; j < matr[0].size(); j++) {
            res[i][j] = matr[i][j] * mul;
        }
    }

    return res;
}

std::vector<std::vector<double>> subMatrix(const std::vector<std::vector<double>> &matr1, const std::vector<std::vector<double>> &matr2) {
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < matr1.size(); i++) {
        res.push_back(std::vector<double>(matr1[0].size(), 0));
    }

    for (size_t i = 0; i < matr1.size(); i++) {
        for (size_t j = 0; j < matr1[0].size(); j++) {
            res[i][j] = matr1[i][j] - matr2[i][j];
        }
    }

    return res;
}

std::vector<std::vector<double>> subMatrixChangingFirstMatrix(std::vector<std::vector<double>> &matr1, const std::vector<std::vector<double>> &matr2) {
    // Не создаем новую матрицу, а меняем первый аргумент
    for (size_t i = 0; i < matr1.size(); i++) {
        for (size_t j = 0; j < matr1[0].size(); j++) {
            matr1[i][j] = matr1[i][j] - matr2[i][j];
        }
    }

    return matr1;
}

std::vector<std::vector<double>> diag(std::vector<double> diagonal) {
    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < diagonal.size(); i++) {
        std::vector<double> line(diagonal.size(), 0);
        line[i] = diagonal[i];
        res.push_back(line);
    }

    return res;
}

std::vector<std::vector<double>> cMDS(const std::vector<std::vector<double>> &matr, std::vector<double> &eigenValuesSqrt, size_t dim=2) {
    std::vector<std::vector<double>> res;
    std::vector<std::vector<double>> eigenVectors;
    std::vector<double> eigensSqrts;
    std::vector<double> eigenVector;
    double eigen = 0;

    std::vector<std::vector<double>> B = matrixSelfTransposedMultiplyParallel(matr);
    std::vector<std::vector<double>> originalB(B);

    for (size_t iterIdx = 0; iterIdx < dim; iterIdx++) {
         eigenVector = powerIterationParallel(B, eigen);

         eigenVectors.push_back(eigenVector);
         eigensSqrts.push_back(sqrt(eigen));

        if (iterIdx < dim - 1) {
            B = subMatrixChangingFirstMatrix(B, vectorSelfTransposedLambda(eigenVector, eigen));
        }
    }

    eigenValuesSqrt = eigensSqrts;

    return matrixTransposedLambdaMultiply(eigenVectors, diag(eigensSqrts));
}

std::vector<std::vector<double>> takePartOfData(std::vector<std::vector<double>> matr, size_t partX, size_t partY) {

    size_t new_m = 0, new_n = 0;

    new_m = (size_t)((double)matr.size() * (double)partX / 10.);
    new_n = (size_t)((double)matr[0].size() * (double)partY / 10.);

    std::vector<std::vector<double>> res;

    for (size_t i = 0; i < new_m; i++) {
        res.push_back(std::vector<double>(new_n, 0));
    }

    for (size_t i = 0; i < new_m; i++) {
        for (size_t j = 0; j < new_n; j++) {
            res[i][j] = matr[i][j];
        }
    }
    // std::cout << new_m << ' ' << new_n << std::endl;
    return res;
}

std::vector<std::vector<double>> evaluateAlgorithm(std::string data, size_t maxStringIdx, size_t maxThreadsPower=10, size_t stableIterCount=5) {
    double start, end, t, t_sum;

    std::vector<std::vector<double>> output;

    std::vector<std::vector<double>> matr = readMatrix(data, maxStringIdx), iterMatr;

    std::cout << "readed" << std::endl;

    std::vector<double> eigenValues;

    for (size_t dataPartY = 1; dataPartY < 11; dataPartY++) {
        for (size_t dataPartX = 1; dataPartX < 11; dataPartX++) {
            std::cout << dataPartX << ' ' << dataPartY << std::endl;

            for (size_t threadsPower = 0; threadsPower < maxThreadsPower + 1; threadsPower++) {
                numThreads = pow(2, threadsPower);

                t_sum = 0;

                for (size_t iterIdx = 0; iterIdx < stableIterCount; iterIdx++) {
                    iterMatr = takePartOfData(matr, dataPartX, dataPartY);

                    start = omp_get_wtime();
                    cMDS(iterMatr, eigenValues, 3);

                    end = omp_get_wtime();
                    t_sum += (end - start);
                }

                t = t_sum / stableIterCount;

                std::vector<double> iterResults {(double)dataPartX, (double)dataPartY, (double)numThreads, t};

                output.push_back(iterResults);
            }
        }
    }

    return output;
}

std::vector<std::vector<double>> evaluateAlgorithmWithoutDataParting(std::string data, size_t maxStringIdx, size_t maxThreadsPower=10, size_t stableIterCount=5) {
    double start, end, t, t_sum;

    std::vector<std::vector<double>> output;

    std::vector<std::vector<double>> matr = readMatrix(data, maxStringIdx), iterMatr;

    std::cout << "readed" << std::endl;

    std::vector<double> eigenValues;

    for (size_t threadsPower = 0; threadsPower < maxThreadsPower + 1; threadsPower++) {
        numThreads = pow(2, threadsPower);

        t_sum = 0;
        std::cout << numThreads << std::endl;

        for (size_t iterIdx = 0; iterIdx < stableIterCount; iterIdx++) {
            std::cout << threadsPower << std::endl;

            start = omp_get_wtime();
            cMDS(matr, eigenValues, 3);

            end = omp_get_wtime();
            t_sum += (end - start);
        }

        t = t_sum / stableIterCount;

        std::vector<double> iterResults {(double)numThreads, t};

        output.push_back(iterResults);
    }



    return output;
}

int main() {
    size_t m = 5000, n = 3000;
    generateRandomCSVMatrix(m, n, "randomMatrix_5k_3k.csv");

    std::vector<std::vector<double>> finalTable = evaluateAlgorithmWithoutDataParting("randomMatrix_5k_3k.csv", -1, 6, 1);

    saveCSVMatrix(finalTable, "randomResults_5k_3k.csv");

    return 0;
}




// Далее идут черновики разных запусков алгоритма.



/*
int main()
{
    size_t m = 5000, n = 30;
    generateRandomCSVMatrix(m, n, "m1.csv");
    std::vector<std::vector<double>> matr = readMatrix("m1.csv");
    // printMatrix<double>(m);

    std::vector<double> v = generateRandomVector(n);
    std::vector<double> vect, vectParallel;
    double start, end, t;

    start = omp_get_wtime();
    vect = matrixVectorMultiply(matr, v);
    end = omp_get_wtime();
    t = (end - start);
    std::cout << "not parallel: " << t << std::endl;

    size_t maxThreadsPower = 10;
    for (size_t threadsPower = 0; threadsPower < maxThreadsPower + 1; threadsPower++) {
        numThreads = pow(2, threadsPower);

        start = omp_get_wtime();
        vectParallel = matrixVectorMultiplyParallel(matr, v);
        end = omp_get_wtime();
        t = (end - start);
        std::cout << "parallel (" << numThreads << "): " << t << std::endl;

        std::cout << "results are equal: " << vectorEquals(vect, vectParallel) << std::endl;
    }
    // printVector(vect);
    // printVector(vectParallel);
//
//    return 0;
//}

*/


/*
int main() {
    size_t m = 100, n = 100;
    generateRandomCSVMatrix(m, n, "randomMatrix.csv", 5, 1);
    std::cout << "gen" << std::endl;

    std::vector<std::vector<double>> finalTable = evaluateAlgorithm("randomMatrix.csv", -1);

    saveCSVMatrix(finalTable, "randomResults.csv");
*/





/*
int main() {

    std::vector<std::vector<double>> matr = readMatrix("m2.csv");
    std::vector<std::vector<double>> vect, vectParallel;
    double start, end, t;

    double mul = 100500.228;

    start = omp_get_wtime();
    vect = matrixScalarMultiply(matr, mul);
    end = omp_get_wtime();
    t = (end - start);
    std::cout << "not parallel: " << t << std::endl;

    size_t maxThreadsPower = 10;
    for (size_t threadsPower = 0; threadsPower < maxThreadsPower + 1; threadsPower++) {
        numThreads = pow(2, threadsPower);

        start = omp_get_wtime();
        vectParallel = matrixScalarMultiplyParallel(matr, mul);
        end = omp_get_wtime();
        t = (end - start);
        std::cout << "parallel (" << numThreads << "): " << t << std::endl;

        std::cout << "results are equal: " << matrixEquals(vect, vectParallel) << std::endl;
    }
*/

    // size_t m = 5000, n = 3000;
    // generateRandomCSVMatrix(m, n, "m2.csv");

/*
    std::vector<std::vector<double>> matr = readMatrix("matrix.csv");

    double eigen1 = 0, eigen2 = 0;
    double start = 0, finish = 0;

    size_t maxThreadsPower = 10;
    for (size_t threadsPower = 0; threadsPower < maxThreadsPower + 1; threadsPower++) {
        numThreads = pow(2, threadsPower);

        start = omp_get_wtime();

        std::vector<std::vector<double>> B = matrixSelfTransposedMultiply(matr);

        std::vector<double> eigenVector = powerIterationParallel(B, eigen1);

        // std::cout << eigen << std::endl;
        // printVector(eigenVector);

        B = subMatrix(B, matrixScalarMultiply(vectorSelfTransposed(eigenVector), eigen1));

        eigenVector = powerIterationParallel(B, eigen2);

        finish = omp_get_wtime();

        std::cout << numThreads * (finish - start) << std::endl;
    }

    std::cout << eigen1 << ' ' << eigen2 << std::endl;
*/
//plan:
// 1. parallel inner product ?
// 2. function for mds
// 3. run it on lomonosov
// 4. wiki + report


/*
    start = omp_get_wtime();
    matrRes = matrixSelfTransposedMultiply(matr);
    end = omp_get_wtime();
    t = (end - start);
    std::cout << "not parallel: " << t << std::endl;

    size_t maxThreadsPower = 10;
    for (size_t threadsPower = 0; threadsPower < maxThreadsPower + 1; threadsPower++) {
        numThreads = pow(2, threadsPower);

        start = omp_get_wtime();
        matrResParallel = matrixSelfTransposedMultiplyParallel(matr);
        end = omp_get_wtime();
        t = (end - start);
        std::cout << "parallel (" << numThreads << "): " << t << std::endl;

        std::cout << "results are equal: " << matrixEquals(matrRes, matrResParallel) << std::endl;
    }
*/
