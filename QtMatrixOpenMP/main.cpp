#include "QtMatrixOpenMP.h"
#include <QtWidgets/QApplication>
#include <Matrix.h>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QtMatrixOpenMP w;
	w.show();
	return a.exec();
}

//#include <iostream>
//#include "Matrix.h"
//#include "MatrixCalculation.h"
//#include <ctime>
//#include "CudaMatrixCalculation.cuh"
//
//#define MATRIXROWA 32
//
//#define MATRIXSAME 32
//
//#define MATRIXCOLB 32
//
//#define MATRIXSIDE 1848
//
//extern "C" Matrix *matrixMulByCuda(Matrix *matrixA, Matrix *matrixB);
//extern "C" int testMulCoreSingleGPU(int coreNum);
//
//int main() {
//	
//	/*clock_t start, ends;
//	start = clock();
//	testMulCoreSingleGPU(8);
//	ends = clock();
//	std::cout << "now end the time, the time is: " << ends - start << std::endl << std::endl;*/
//	clock_t start, ends;
//
//	auto *aa = new Matrix();
//	auto *bb = new Matrix();
//	aa->randomMatrix(MATRIXROWA, MATRIXSAME, FLOAT, 1, 6);
//	bb->randomMatrix(MATRIXSAME, MATRIXCOLB, LONGLONG, 1, 6);
//
//	start = clock();
//	std::cout << "now start time of normal" << std::endl;
//	Matrix *cc = MatrixCalculation::matrixMul(aa, bb);
//	ends = clock();
//	std::cout << "now end the time of normal, the time is: " << ends - start << std::endl << std::endl;
//
//
//	start = clock();
//	std::cout << "now start time of CudaNormal" << std::endl;
//	Matrix *rr = matrixMulByCuda(aa, bb);
//	ends = clock();
//	std::cout << "the answer of cuda normal is: " << cc->matrixCompare(rr) << std::endl;
//	std::cout << "now end the time of cuda normal, the time is: " << ends - start << std::endl << std::endl;
//	rr->writeMatrix("test.txt");
//
//	/*start = clock();
//	std::cout << "now start time of test" << std::endl;
//	Matrix *hh = MatrixCalculation::algorithmStrassen(aa, bb, 4, NORMALMATRIXMUL);
//	ends = clock();
//	std::cout << "the answer of test is: " << cc->matrixCompare(hh) << std::endl;
//	std::cout << "now end the time of test, the time is: " << ends - start << std::endl << std::endl;
//
//
//	start = clock();
//	std::cout << "now start time of DNS" << std::endl;
//	Matrix *kk = MatrixCalculation::algorithmDNS(aa, bb, 8, NORMALMATRIXMUL);
//	ends = clock();
//	std::cout << "the answer of DNS is: " << cc->matrixCompare(kk) << std::endl;
//	std::cout << "now end the time of DNS, the time is: " << ends - start << std::endl << std::endl;
//
//
//	start = clock();
//	std::cout << "now start time of normal parallel" << std::endl;
//	Matrix *ff = MatrixCalculation::matrixMulParallel(aa, bb, 8);
//	ends = clock();
//	std::cout << "the answer of normal parallel is: " << cc->matrixCompare(ff) << std::endl;
//	std::cout << "now end the time of normal parallel, the time is: " << ends - start << std::endl << std::endl;
//
//
//	start = clock();
//	std::cout << "now start time of Strassen" << std::endl;
//	Matrix *dd = MatrixCalculation::algorithmStrassen(aa, bb, 0, ALGOSTRASSEN);
//	ends = clock();
//	std::cout << "the answer of Strassen is: " << cc->matrixCompare(dd) << std::endl;
//	std::cout << "now end the time of Strasseen, the time is: " << ends - start << std::endl << std::endl;
//
//
//	start = clock();
//	std::cout << "now start time of Strassen parallel" << std::endl;
//	Matrix *gg = MatrixCalculation::algorithmStrassen(aa, bb, 8, ALGOSTRASSEN);
//	ends = clock();
//	std::cout << "the answer of Strassen is: " << cc->matrixCompare(gg) << std::endl;
//	std::cout << "now end the time of Strasseen parallel, the time is: " << ends - start << std::endl << std::endl;
//
//
//	start = clock();
//	std::cout << "now start time of Cannon" << std::endl;
//	Matrix *ee = MatrixCalculation::algorithmCannon(aa, bb, 9, NORMALMATRIXMUL);
//	ends = clock();
//	std::cout << "the answer of Cannon is: " << cc->matrixCompare(ee) << std::endl;
//	std::cout << "now end the time of Cannon, the time is: " << ends - start << std::endl << std::endl;*/
//
//	system("pause");
//}


