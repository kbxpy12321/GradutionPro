//
// Created by LZR on 2019/2/2.
//

#include <iostream>
#include "MatrixCalculation.h"
#include "Matrix.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>

Matrix *MatrixCalculation::matrixAdd(Matrix *leftMatrix, Matrix *rightMatrix) {
	if (leftMatrix->getRow() != rightMatrix->getRow() || leftMatrix->getCol() != rightMatrix->getCol()) {
		return nullptr;
	}
	int matrixType = matrixTypeDecision(leftMatrix->getType(), rightMatrix->getType());
	auto *tmpMatrix = new Matrix(leftMatrix->getRow(), leftMatrix->getCol(), matrixType);
	int len = leftMatrix->getCol() * leftMatrix->getRow();
	for (int i = 0; i < len; i++) {
		tmpMatrix->matrixPush(leftMatrix->getMatrixElement(i) + rightMatrix->getMatrixElement(i));
	}
	return tmpMatrix;
}
Matrix *MatrixCalculation::matrixSub(Matrix *leftMatrix, Matrix *rightMatrix) {
	if (leftMatrix->getRow() != rightMatrix->getRow() || leftMatrix->getCol() != rightMatrix->getCol()) {
		return nullptr;
	}
	int matrixType = matrixTypeDecision(leftMatrix->getType(), rightMatrix->getType());
	auto *tmpMatrix = new Matrix(leftMatrix->getRow(), leftMatrix->getCol(), matrixType);
	int len = leftMatrix->getCol() * leftMatrix->getRow();
	for (int i = 0; i < len; i++) {
		tmpMatrix->matrixPush(leftMatrix->getMatrixElement(i) - rightMatrix->getMatrixElement(i));
	}
	return tmpMatrix;
}

Matrix *MatrixCalculation::matrixMul(Matrix *leftMatrix, Matrix *rightMatrix) {
	if (leftMatrix->getCol() != rightMatrix->getRow())
		return nullptr;
	int leftRow = leftMatrix->getRow();
	int col = leftMatrix->getCol();
	int rightCol = rightMatrix->getCol();
	int matrixType = matrixTypeDecision(leftMatrix->getType(), rightMatrix->getType());
	auto *tmpMatrix = new Matrix(leftRow, rightCol, matrixType);
	double sumPerLine = 0;
	for (int i = 1; i <= leftRow; i++) {
		for (int k = 1; k <= rightCol; k++) {
			for (int j = 1; j <= col; j++) {
				sumPerLine += leftMatrix->getMatrixElement(i, j) * rightMatrix->getMatrixElement(j, k);
			}
			tmpMatrix->matrixPush(sumPerLine);
			sumPerLine = 0;
		}
	}
	return tmpMatrix;
}

Matrix *MatrixCalculation::matrixMulParallel(Matrix *leftMatrix, Matrix *rightMatrix, int coreNum) {
	if (leftMatrix->getCol() != rightMatrix->getRow())
		return nullptr;
	int leftRow = leftMatrix->getRow();
	int col = leftMatrix->getCol();
	int rightCol = rightMatrix->getCol();
	int matrixType = matrixTypeDecision(leftMatrix->getType(), rightMatrix->getType());
	auto *tmpMatrix = new Matrix(leftRow, rightCol, matrixType);
	tmpMatrix->initVectorSpace();
	double sumPerLine = 0;
	omp_set_num_threads(coreNum);
	int i, j, k;
#pragma omp parallel for private(j, k) firstprivate(sumPerLine)
	for (i = 1; i <= leftRow; i++) {
		for (k = 1; k <= rightCol; k++) {
			for (j = 1; j <= col; j++) {
				sumPerLine += leftMatrix->getMatrixElement(i, j) * rightMatrix->getMatrixElement(j, k);
			}
			tmpMatrix->setMatrixElement(i, k, sumPerLine);
			sumPerLine = 0;
		}
	}
	return tmpMatrix;
}

Matrix *MatrixCalculation::expandMatrixWithZero(Matrix *outMatrix, int newRow, int newCol) {
	auto *resMatrix = new Matrix(newRow, newCol, outMatrix->getType());
	resMatrix->initVectorSpace();
#pragma omp parallel for
	for (int i = 1; i <= outMatrix->getRow(); i++) {
		for (int j = 1; j <= outMatrix->getCol(); j++) {
			resMatrix->setMatrixElement(i, j, outMatrix->getMatrixElement(i, j));
		}
	}
	return resMatrix;
}

Matrix *MatrixCalculation::algorithmStrassen(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCode) {
	if (matrixA->getCol() != matrixB->getRow())
		return nullptr;
	int row = matrixA->getRow();
	int col = matrixB->getCol();
	int sameSide = matrixA->getCol();
	int maxSide = row;
	if (col > maxSide)
		maxSide = col;
	if (sameSide > maxSide)
		maxSide = sameSide;

	if (maxSide < 64) {
		std::cout << "abandoned Strassen because it's not as fast as normal algo here" << std::endl;
		return MatrixCalculation::matrixMul(matrixA, matrixB);
	}

	int towPower = 2;
	while (maxSide / towPower > 64) {
		towPower *= 2;
	}
	if (maxSide % towPower != 0) {
		maxSide = maxSide / towPower + 1;
		maxSide *= towPower;
	}
	else if (row == col && col == sameSide) {
		if (coreNum <= 1)
			return Strassen(matrixA, matrixB);
		return StrassenParallel(matrixA, matrixB, coreNum, mulCode);
	}


	if (pow(maxSide, 2.81) > (double)row * col * sameSide) {
		std::cout << "abandoned Strassen because it's not as fast as normal algo here" << std::endl;
		if (coreNum <= 1)
			return matrixMul(matrixA, matrixB);
		return MatrixCalculation::matrixMulParallel(matrixA, matrixB, coreNum);
	}

	Matrix *tmpMatrixA = expandMatrixWithZero(matrixA, maxSide, maxSide);
	Matrix *tmpMatrixB = expandMatrixWithZero(matrixB, maxSide, maxSide);
	Matrix *tmpResMatrix;

	if (coreNum <= 1) {
		tmpResMatrix = Strassen(tmpMatrixA, tmpMatrixB);
	}
	else {
		tmpResMatrix = StrassenParallel(tmpMatrixA, tmpMatrixB, coreNum, mulCode);
	}

	Matrix *resMatrix = tmpResMatrix->generateMatrixParts(1, 1, row, col);

	delete(tmpMatrixA);
	delete(tmpMatrixB);
	delete(tmpResMatrix);

	return resMatrix;
}


Matrix *MatrixCalculation::StrassenParallel(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCode) {
	int sideLen = matrixA->getRow();
	int halfLen = sideLen / 2;
	int matrixType = matrixTypeDecision(matrixA->getType(), matrixB->getType());
	if (halfLen < 64) {
		return MatrixCalculation::matrixMulParallel(matrixA, matrixB, coreNum);
	}

	omp_set_num_threads(coreNum);

	auto *A11 = new Matrix(halfLen, halfLen, matrixType);
	auto *A12 = new Matrix(halfLen, halfLen, matrixType);
	auto *A21 = new Matrix(halfLen, halfLen, matrixType);
	auto *A22 = new Matrix(halfLen, halfLen, matrixType);
	auto *B11 = new Matrix(halfLen, halfLen, matrixType);
	auto *B12 = new Matrix(halfLen, halfLen, matrixType);
	auto *B21 = new Matrix(halfLen, halfLen, matrixType);
	auto *B22 = new Matrix(halfLen, halfLen, matrixType);

	for (int i = 1; i <= halfLen; i++) {
		for (int j = 1; j <= halfLen; j++) {
			A11->matrixPush(matrixA->getMatrixElement(i, j));
			A12->matrixPush(matrixA->getMatrixElement(i, j + halfLen));
			A21->matrixPush(matrixA->getMatrixElement(i + halfLen, j));
			A22->matrixPush(matrixA->getMatrixElement(i + halfLen, j + halfLen));

			B11->matrixPush(matrixB->getMatrixElement(i, j));
			B12->matrixPush(matrixB->getMatrixElement(i, j + halfLen));
			B21->matrixPush(matrixB->getMatrixElement(i + halfLen, j));
			B22->matrixPush(matrixB->getMatrixElement(i + halfLen, j + halfLen));
		}
	}

	//printf("now into the parallel");

//    A11->printMatrix();
//    std::cout<<std::endl;
//    A12->printMatrix();
//    std::cout<<std::endl;
//    A21->printMatrix();
//    std::cout<<std::endl;
//    A22->printMatrix();
//    std::cout<<std::endl;
//    B11->printMatrix();
//    std::cout<<std::endl;
//    B12->printMatrix();
//    std::cout<<std::endl;
//    B21->printMatrix();
//    std::cout<<std::endl;
//    B22->printMatrix();
//    std::cout<<std::endl;



	Matrix *M1, *M2, *M3, *M4, *M5, *M6, *M7;
	int coreNumLeft = omp_get_num_threads() - coreNum;
#pragma omp parallel
	{
#pragma omp sections nowait
		{
#pragma omp section
			{
				//printf("M1:%d", omp_get_thread_num());
				if (mulCode == NORMALMATRIXMUL) {
					M1 = matrixMul(MatrixCalculation::matrixAdd(A11, A22), MatrixCalculation::matrixAdd(B11, B22));
				}
				else if (mulCode == NORMALMATRIXMULPARALLEL) {
					M1 = matrixMulParallel(MatrixCalculation::matrixAdd(A11, A22), MatrixCalculation::matrixAdd(B11, B22), coreNumLeft);
				}
				else{
					M1 = Strassen(MatrixCalculation::matrixAdd(A11, A22), MatrixCalculation::matrixAdd(B11, B22));
				}
			
			}
#pragma omp section
			{
				//printf("M2:%d", omp_get_thread_num());
				if (mulCode == NORMALMATRIXMUL) {
					M2 = matrixMul(MatrixCalculation::matrixAdd(A21, A22), B11);
				}
				else if (mulCode == NORMALMATRIXMULPARALLEL) {
					M2 = matrixMulParallel(MatrixCalculation::matrixAdd(A21, A22), B11, coreNumLeft);
				}
				else {
					M2 = Strassen(MatrixCalculation::matrixAdd(A21, A22), B11);
				}
			}
#pragma omp section
			{
				//printf("M3:%d", omp_get_thread_num());
				if (mulCode == NORMALMATRIXMUL) {
					M3 = matrixMul(A11, MatrixCalculation::matrixSub(B12, B22));
				}
				else if (mulCode == NORMALMATRIXMULPARALLEL) {
					M3 = matrixMulParallel(A11, MatrixCalculation::matrixSub(B12, B22), coreNumLeft);
				}
				else {
					M3 = Strassen(A11, MatrixCalculation::matrixSub(B12, B22));
				}
				
			}
#pragma omp section
			{
				//printf("M4:%d", omp_get_thread_num());
				if (mulCode == NORMALMATRIXMUL) {
					M4 = matrixMul(A22, MatrixCalculation::matrixSub(B21, B11));
				}
				else if (mulCode == NORMALMATRIXMULPARALLEL) {
					M4 = matrixMulParallel(A22, MatrixCalculation::matrixSub(B21, B11), coreNumLeft);
				}
				else {
					M4 = Strassen(A22, MatrixCalculation::matrixSub(B21, B11));
				}
			}
#pragma omp section
			{
				//printf("M5:%d", omp_get_thread_num());
				if (mulCode == NORMALMATRIXMUL) {
					M5 = matrixMul(MatrixCalculation::matrixAdd(A11, A12), B22);
				}
				else if (mulCode == NORMALMATRIXMULPARALLEL) {
					M5 = matrixMulParallel(MatrixCalculation::matrixAdd(A11, A12), B22, coreNumLeft);
				}
				else {
					M5 = Strassen(MatrixCalculation::matrixAdd(A11, A12), B22);
				}
			}
#pragma omp section
			{
				//printf("M6:%d", omp_get_thread_num());
				if (mulCode == NORMALMATRIXMUL) {
					M6 = matrixMul(MatrixCalculation::matrixSub(A21, A11), MatrixCalculation::matrixAdd(B11, B12));
				}
				else if (mulCode == NORMALMATRIXMULPARALLEL) {
					M6 = matrixMulParallel(MatrixCalculation::matrixSub(A21, A11), MatrixCalculation::matrixAdd(B11, B12), coreNumLeft);
				}
				else {
					M6 = Strassen(MatrixCalculation::matrixSub(A21, A11), MatrixCalculation::matrixAdd(B11, B12));
				}
			}
#pragma omp section
			{
				//printf("M7:%d", omp_get_thread_num());
				if (mulCode == NORMALMATRIXMUL) {
					M7 = matrixMul(MatrixCalculation::matrixSub(A12, A22), MatrixCalculation::matrixAdd(B21, B22));
				}
				else if (mulCode == NORMALMATRIXMULPARALLEL) {
					M7 = matrixMulParallel(MatrixCalculation::matrixSub(A12, A22), MatrixCalculation::matrixAdd(B21, B22), coreNumLeft);
				}
				else {
					M7 = Strassen(MatrixCalculation::matrixSub(A12, A22), MatrixCalculation::matrixAdd(B21, B22));
				}
				
			}
		}
	};
	//    M1->printMatrix();
	//    std::cout<<std::endl;
	//    M2->printMatrix();
	//    std::cout<<std::endl;
	//    M3->printMatrix();
	//    std::cout<<std::endl;
	//    M4->printMatrix();
	//    std::cout<<std::endl;
	//    M5->printMatrix();
	//    std::cout<<std::endl;
	//    M6->printMatrix();
	//    std::cout<<std::endl;
	//    M7->printMatrix();
	//    std::cout<<std::endl;

	auto* resultMatrix = new Matrix(sideLen, sideLen, matrixType);

	int x = 1, y = 1;

	for (int i = 1; i <= sideLen; i++) {
		for (int j = 1; j <= sideLen; j++) {
			if (i <= halfLen && j <= halfLen) {
				x = i;
				y = j;
				resultMatrix->matrixPush(M1->getMatrixElement(x, y) + M4->getMatrixElement(x, y)
					- M5->getMatrixElement(x, y) + M7->getMatrixElement(x, y));
			}
			else if (i <= halfLen && j > halfLen) {
				x = i;
				y = j - halfLen;
				resultMatrix->matrixPush(M3->getMatrixElement(x, y) + M5->getMatrixElement(x, y));
			}
			else if (i > halfLen && j <= halfLen) {
				x = i - halfLen;
				y = j;
				resultMatrix->matrixPush(M2->getMatrixElement(x, y) + M4->getMatrixElement(x, y));
			}
			else if (i > halfLen && j > halfLen) {
				x = i - halfLen;
				y = j - halfLen;
				resultMatrix->matrixPush(M1->getMatrixElement(x, y) - M2->getMatrixElement(x, y)
					+ M3->getMatrixElement(x, y) + M6->getMatrixElement(x, y));
			}
		}
	}

	delete(A11);
	delete(A12);
	delete(A21);
	delete(A22);
	delete(B11);
	delete(B12);
	delete(B21);
	delete(B22);
	delete(M1);
	delete(M2);
	delete(M3);
	delete(M4);
	delete(M5);
	delete(M6);
	delete(M7);
	//    delete(matrixA);
	//    delete(matrixB);

	return resultMatrix;
}

Matrix* MatrixCalculation::Strassen(Matrix *matrixA, Matrix *matrixB) {
	int sideLen = matrixA->getRow();
	int halfLen = sideLen / 2;
	int matrixType = matrixTypeDecision(matrixA->getType(), matrixB->getType());
	if (halfLen < 64) {
		return MatrixCalculation::matrixMul(matrixA, matrixB);
	}

	auto *A11 = new Matrix(halfLen, halfLen, matrixType);
	auto *A12 = new Matrix(halfLen, halfLen, matrixType);
	auto *A21 = new Matrix(halfLen, halfLen, matrixType);
	auto *A22 = new Matrix(halfLen, halfLen, matrixType);
	auto *B11 = new Matrix(halfLen, halfLen, matrixType);
	auto *B12 = new Matrix(halfLen, halfLen, matrixType);
	auto *B21 = new Matrix(halfLen, halfLen, matrixType);
	auto *B22 = new Matrix(halfLen, halfLen, matrixType);

	for (int i = 1; i <= halfLen; i++) {
		for (int j = 1; j <= halfLen; j++) {
			A11->matrixPush(matrixA->getMatrixElement(i, j));
			A12->matrixPush(matrixA->getMatrixElement(i, j + halfLen));
			A21->matrixPush(matrixA->getMatrixElement(i + halfLen, j));
			A22->matrixPush(matrixA->getMatrixElement(i + halfLen, j + halfLen));

			B11->matrixPush(matrixB->getMatrixElement(i, j));
			B12->matrixPush(matrixB->getMatrixElement(i, j + halfLen));
			B21->matrixPush(matrixB->getMatrixElement(i + halfLen, j));
			B22->matrixPush(matrixB->getMatrixElement(i + halfLen, j + halfLen));
		}
	}

	//    A11->printMatrix();
	//    std::cout<<std::endl;
	//    A12->printMatrix();
	//    std::cout<<std::endl;
	//    A21->printMatrix();
	//    std::cout<<std::endl;
	//    A22->printMatrix();
	//    std::cout<<std::endl;
	//    B11->printMatrix();
	//    std::cout<<std::endl;
	//    B12->printMatrix();
	//    std::cout<<std::endl;
	//    B21->printMatrix();
	//    std::cout<<std::endl;
	//    B22->printMatrix();
	//    std::cout<<std::endl;



	Matrix *M1 = Strassen(MatrixCalculation::matrixAdd(A11, A22), MatrixCalculation::matrixAdd(B11, B22));
	Matrix *M2 = Strassen(MatrixCalculation::matrixAdd(A21, A22), B11);
	Matrix *M3 = Strassen(A11, MatrixCalculation::matrixSub(B12, B22));
	Matrix *M4 = Strassen(A22, MatrixCalculation::matrixSub(B21, B11));
	Matrix *M5 = Strassen(MatrixCalculation::matrixAdd(A11, A12), B22);
	Matrix *M6 = Strassen(MatrixCalculation::matrixSub(A21, A11), MatrixCalculation::matrixAdd(B11, B12));
	Matrix *M7 = Strassen(MatrixCalculation::matrixSub(A12, A22), MatrixCalculation::matrixAdd(B21, B22));

	//    M1->printMatrix();
	//    std::cout<<std::endl;
	//    M2->printMatrix();
	//    std::cout<<std::endl;
	//    M3->printMatrix();
	//    std::cout<<std::endl;
	//    M4->printMatrix();
	//    std::cout<<std::endl;
	//    M5->printMatrix();
	//    std::cout<<std::endl;
	//    M6->printMatrix();
	//    std::cout<<std::endl;
	//    M7->printMatrix();
	//    std::cout<<std::endl;

	auto* resultMatrix = new Matrix(sideLen, sideLen, matrixType);

	int x = 1, y = 1;

	for (int i = 1; i <= sideLen; i++) {
		for (int j = 1; j <= sideLen; j++) {
			if (i <= halfLen && j <= halfLen) {
				x = i;
				y = j;
				resultMatrix->matrixPush(M1->getMatrixElement(x, y) + M4->getMatrixElement(x, y)
					- M5->getMatrixElement(x, y) + M7->getMatrixElement(x, y));
			}
			else if (i <= halfLen && j > halfLen) {
				x = i;
				y = j - halfLen;
				resultMatrix->matrixPush(M3->getMatrixElement(x, y) + M5->getMatrixElement(x, y));
			}
			else if (i > halfLen && j <= halfLen) {
				x = i - halfLen;
				y = j;
				resultMatrix->matrixPush(M2->getMatrixElement(x, y) + M4->getMatrixElement(x, y));
			}
			else if (i > halfLen && j > halfLen) {
				x = i - halfLen;
				y = j - halfLen;
				resultMatrix->matrixPush(M1->getMatrixElement(x, y) - M2->getMatrixElement(x, y)
					+ M3->getMatrixElement(x, y) + M6->getMatrixElement(x, y));
			}
		}
	}

	delete(A11);
	delete(A12);
	delete(A21);
	delete(A22);
	delete(B11);
	delete(B12);
	delete(B21);
	delete(B22);
	delete(M1);
	delete(M2);
	delete(M3);
	delete(M4);
	delete(M5);
	delete(M6);
	delete(M7);
	//    delete(matrixA);
	//    delete(matrixB);

	return resultMatrix;
}

int CoreDivision(int coreNum) {
	switch (coreNum) {
	case 1:
		return 1;
	case 2:
	case 3:
	case 4:
		return 2;
	case 5:
	case 6:
	case 7:
	case 8:
	case 9:
		return 3;
	default:
		return 4;
	}
}

int ThreadDivision(int coreNum) {
	if (coreNum <= 1)
		return 1;
	if (coreNum <= 8)
		return 2;
	if (coreNum <= 27)
		return 3;
	if (coreNum <= 64)
		return 4;
	return 5;
}

Matrix *MatrixCalculation::algorithmDNS(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCore) {
	if (matrixA->getCol() != matrixB->getRow())
		return nullptr;

	if (coreNum <= 1)
		return MatrixCalculation::matrixMulParallel(matrixA, matrixB, coreNum);

	int threadLenDivision = ThreadDivision(coreNum);

	int matrixARowModDivision = (threadLenDivision - matrixA->getRow() % threadLenDivision) % threadLenDivision;
	int matrixBColModDivision = (threadLenDivision - matrixB->getCol() % threadLenDivision) % threadLenDivision;
	int matrixABSameSideModDivision = (threadLenDivision - matrixA->getCol() % threadLenDivision) % threadLenDivision;
	if (matrixARowModDivision == 0 && matrixBColModDivision == 0 && matrixABSameSideModDivision == 0)
		return DNS(matrixA, matrixB, threadLenDivision, mulCore);
	else if (matrixARowModDivision == 0 && matrixABSameSideModDivision == 0) {
		Matrix *tmpMatrixB = expandMatrixWithZero(matrixB, matrixB->getRow() + matrixABSameSideModDivision, matrixB->getCol() + matrixBColModDivision);
		Matrix *tmpResMatrix = DNS(matrixA, tmpMatrixB, threadLenDivision, mulCore);
		Matrix *resMatrix = tmpResMatrix->generateMatrixParts(1, 1, matrixA->getRow(), matrixB->getCol());
		delete (tmpMatrixB);
		delete (tmpResMatrix);
		return resMatrix;
	}
	else if (matrixBColModDivision == 0 && matrixABSameSideModDivision == 0) {
		Matrix *tmpMatrixA = expandMatrixWithZero(matrixA, matrixA->getRow() + matrixARowModDivision, matrixA->getCol() + matrixABSameSideModDivision);
		Matrix *tmpResMatrix = DNS(tmpMatrixA, matrixB, threadLenDivision, mulCore);
		Matrix *resMatrix = tmpResMatrix->generateMatrixParts(1, 1, matrixA->getRow(), matrixB->getCol());

		delete (tmpMatrixA);
		delete (tmpResMatrix);

		return resMatrix;
	}
	else {
		Matrix *tmpMatrixA = expandMatrixWithZero(matrixA, matrixA->getRow() + matrixARowModDivision, matrixA->getCol() + matrixABSameSideModDivision);
		Matrix *tmpMatrixB = expandMatrixWithZero(matrixB, matrixB->getRow() + matrixABSameSideModDivision, matrixB->getCol() + matrixBColModDivision);
		Matrix *tmpResMatrix = DNS(tmpMatrixA, tmpMatrixB, threadLenDivision, mulCore);
		Matrix *resMatrix = tmpResMatrix->generateMatrixParts(1, 1, matrixA->getRow(), matrixB->getCol());
		delete (tmpMatrixA);
		delete (tmpMatrixB);
		delete (tmpResMatrix);
		return resMatrix;
	}
}

Matrix *MatrixCalculation::DNS(Matrix *matrixA, Matrix *matrixB, int threadLenDivision, int mulCode) {
	int tid;
	int tidI, tidJ, tidK;
	int threadSquareDivision = threadLenDivision * threadLenDivision;
	int threadCubeDivision = threadLenDivision * threadLenDivision * threadLenDivision;
	omp_set_num_threads(threadCubeDivision);

	auto **tmpResMatrix = new Matrix *[threadLenDivision];
	for (int i = 0; i < threadLenDivision; i++) {
		tmpResMatrix[i] = new Matrix(matrixA->getRow(), matrixB->getCol(), matrixTypeDecision(matrixA->getType(), matrixB->getType()));
		tmpResMatrix[i]->initVectorSpace();
	}
	auto *resMatrix = new Matrix(matrixA->getRow(), matrixB->getCol(), matrixTypeDecision(matrixA->getType(), matrixB->getType()));
#pragma omp parallel private(tid)
	{
		//tid = 0;
		tid = omp_get_thread_num();
		tidK = tid % threadLenDivision;
		tidJ = (tid - tidK) % threadSquareDivision / threadLenDivision;
		tidI = (tid - tidK - tidJ * threadLenDivision) % threadCubeDivision / threadSquareDivision;

		int blockIdA, blockIdB;
		blockIdA = tidI + tidJ * threadLenDivision;
		blockIdB = tidI * threadLenDivision + tidK;
		if (mulCode == NORMALMATRIXMUL) {
			matrixMulAndInsertByBlock(matrixA, matrixB, tmpResMatrix[tidI], blockIdA, blockIdB, threadLenDivision, 1);
		}
		else if (mulCode == NORMALMATRIXMULPARALLEL) {
			matrixMulAndInsertByBlock(matrixA, matrixB, tmpResMatrix[tidI], blockIdA, blockIdB, threadLenDivision, 99999);
		}
		else {
			//TODO
		}
	}

	int fullLen = resMatrix->getRow() * resMatrix->getCol();
	double tmpSum;
	for (int i = 0; i < fullLen; i++) {
		tmpSum = 0;
		for (int j = 0; j < threadLenDivision; j++) {
			tmpSum += tmpResMatrix[j]->getMatrixElement(i);
		}
		resMatrix->matrixPush(tmpSum);
	}
	for (int j = 0; j < threadLenDivision; j++) {
		delete(tmpResMatrix[j]);
	}
	delete[]tmpResMatrix;
	return resMatrix;
}

Matrix *MatrixCalculation::algorithmCannon(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCode) {
	if (matrixA->getCol() != matrixB->getRow())
		return nullptr;
	int matrixSideDivision = CoreDivision(coreNum);
	int matrixARowModDivision = (matrixSideDivision - matrixA->getRow() % matrixSideDivision) % matrixSideDivision;
	int matrixBColModDivision = (matrixSideDivision - matrixB->getCol() % matrixSideDivision) % matrixSideDivision;
	int matrixABSameSideModDivision = (matrixSideDivision - matrixA->getCol() % matrixSideDivision) % matrixSideDivision;
	if (matrixARowModDivision == 0 && matrixBColModDivision == 0 && matrixABSameSideModDivision == 0)
		return Cannon(matrixA, matrixB, matrixSideDivision, mulCode);
	else if (matrixARowModDivision == 0 && matrixABSameSideModDivision == 0) {
		Matrix *tmpMatrixB = expandMatrixWithZero(matrixB, matrixB->getRow() + matrixABSameSideModDivision, matrixB->getCol() + matrixBColModDivision);

		Matrix *tmpResMatrix = Cannon(matrixA, tmpMatrixB, matrixSideDivision, mulCode);
		Matrix *resMatrix = tmpResMatrix->generateMatrixParts(1, 1, matrixA->getRow(), matrixB->getCol());
		delete (tmpMatrixB);
		delete (tmpResMatrix);
		return resMatrix;
	}
	else if (matrixBColModDivision == 0 && matrixABSameSideModDivision == 0) {
		Matrix *tmpMatrixA = expandMatrixWithZero(matrixA, matrixA->getRow() + matrixARowModDivision, matrixA->getCol() + matrixABSameSideModDivision);
		Matrix *tmpResMatrix = Cannon(tmpMatrixA, matrixB, matrixSideDivision, mulCode);
		Matrix *resMatrix = tmpResMatrix->generateMatrixParts(1, 1, matrixA->getRow(), matrixB->getCol());

		delete (tmpMatrixA);
		delete (tmpResMatrix);

		return resMatrix;
	}
	else {
		Matrix *tmpMatrixA = expandMatrixWithZero(matrixA, matrixA->getRow() + matrixARowModDivision, matrixA->getCol() + matrixABSameSideModDivision);
		Matrix *tmpMatrixB = expandMatrixWithZero(matrixB, matrixB->getRow() + matrixABSameSideModDivision, matrixB->getCol() + matrixBColModDivision);

		Matrix *tmpResMatrix = Cannon(tmpMatrixA, tmpMatrixB, matrixSideDivision, mulCode);
		Matrix *resMatrix = tmpResMatrix->generateMatrixParts(1, 1, matrixA->getRow(), matrixB->getCol());

		delete (tmpMatrixA);
		delete (tmpMatrixB);
		delete (tmpResMatrix);

		return resMatrix;
	}
}

Matrix *MatrixCalculation::Cannon(Matrix *matrixA, Matrix *matrixB, int matrixSideDivision, int mulCode) {
	omp_set_num_threads(matrixSideDivision * matrixSideDivision);
	int tid;
	auto *resMatrix = new Matrix(matrixA->getRow(), matrixB->getCol(), matrixTypeDecision(matrixA->getType(), matrixB->getType()));
	resMatrix->initVectorSpace();
#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num();
		//tid = 2;
		//std::cout<<"now print the thread tid:"<<tid<<std::endl;
		int rowA = tid / matrixSideDivision;
		int colB = tid % matrixSideDivision;
		int tidA = tid - colB;
		if (tidA < rowA * matrixSideDivision)
			tidA += matrixSideDivision;
		int tidB = tid - rowA * matrixSideDivision;
		if (tidB < colB)
			tidB += matrixSideDivision * matrixSideDivision;

		for (int i = 0; i < matrixSideDivision; i++) {

			if (mulCode == NORMALMATRIXMUL) {
				matrixMulAndInsertByBlock(matrixA, matrixB, resMatrix, tidA, tidB, matrixSideDivision, 1);
			}
			else if (mulCode == NORMALMATRIXMULPARALLEL) {
				matrixMulAndInsertByBlock(matrixA, matrixB, resMatrix, tidA, tidB, matrixSideDivision, 99999);
			}
			else {
				//TODO
			}
			tidA--;
			tidB -= matrixSideDivision;
			if (tidA < rowA * matrixSideDivision)
				tidA += matrixSideDivision;
			if (tidB < colB)
				tidB += matrixSideDivision * matrixSideDivision;

			//if(tid == 0)
			//std::cout<<"in thread; "<<tid<<" print tidA:"<<tidA<<" print tidB:"<<tidB<<std::endl;
		}
	}
	return resMatrix;
}

Matrix *MatrixCalculation::matrixMulAndInsertByBlock(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC, int blockIdA, int blockIdB, int sideDivision, int coreNum) {
	int allCores = omp_get_num_threads();
	if (coreNum > allCores)
		coreNum = allCores + 1;
	omp_set_num_threads(coreNum);
	int blockRowA = matrixA->getRow() / sideDivision;
	int blockColA = matrixA->getCol() / sideDivision;
	int blockRowB = matrixB->getRow() / sideDivision;
	int blockColB = matrixB->getCol() / sideDivision;
	int matrixALeftTopX = (blockIdA / sideDivision) * blockRowA + 1;
	int matrixALeftTopY = (blockIdA % sideDivision) * blockColA + 1;
	int matrixBLeftTopX = (blockIdB / sideDivision) * blockRowB + 1;
	int matrixBLeftTopY = (blockIdB % sideDivision) * blockColB + 1;
	int matrixARightDownX = matrixALeftTopX + blockRowA - 1;
	int matrixARightDownY = matrixALeftTopY + blockColA - 1;
	int matrixBRightDownX = matrixBLeftTopX + blockRowB - 1;
	int matrixBRightDownY = matrixBLeftTopY + blockColB - 1;
	double sumPerLine = 0;
	int sameSideLen = matrixARightDownY - matrixALeftTopY + 1;
	int startAY, startBX;
#pragma omp parallel for if(coreNum > allCores) schedule(guided)
	for (int i = matrixALeftTopX; i <= matrixARightDownX; i++) {
		for (int k = matrixBLeftTopY; k <= matrixBRightDownY; k++) {
			startAY = matrixALeftTopY;
			startBX = matrixBLeftTopX;
			for (int j = 0; j < sameSideLen; j++) {
				sumPerLine += matrixA->getMatrixElement(i, startAY) * matrixB->getMatrixElement(startBX, k);
				startAY++;
				startBX++;
			}
			matrixC->addIntoMatrixElement(i, k, sumPerLine);
			sumPerLine = 0;
		}
	}
	return nullptr;
}

int MatrixCalculation::matrixTypeDecision(int typeA, int typeB) {
	if (typeA == typeB)
		return typeA;
	if (typeA == DOUBLE || typeB == DOUBLE)
		return DOUBLE;
	if ((typeA == LONGLONG && typeB == FLOAT) || (typeB == LONGLONG && typeA == FLOAT))
		return DOUBLE;
	if (typeA == LONGLONG || typeB == LONGLONG)
		return LONGLONG;
	if (typeA == FLOAT || typeB == FLOAT)
		return FLOAT;
	return INTEGER;
}




