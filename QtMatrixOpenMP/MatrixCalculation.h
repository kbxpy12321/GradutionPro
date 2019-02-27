//
// Created by LZR on 2019/2/2.
//
#pragma once

#include "Matrix.h"

class MatrixCalculation {
public:
	static Matrix* matrixAdd(Matrix *leftMatrix, Matrix *rightMatrix);
	static Matrix* matrixSub(Matrix *leftMatrix, Matrix *rightMatrix);
	static Matrix* matrixMul(Matrix *leftMatrix, Matrix *rightMatrix);
	static Matrix* matrixMulParallel(Matrix *leftMatrix, Matrix *rightMatrix, int coreNum);

	static Matrix* Strassen(Matrix *matrixA, Matrix *matrixB);
	static Matrix* algorithmStrassen(Matrix *matrixA, Matrix *matrixB, int coreNum);
	static Matrix* StrassenParallel(Matrix *matrixA, Matrix *matrixB, int coreNum);

	static Matrix* algorithmCannon(Matrix *matrixA, Matrix *matrixB, int coreNum);
	static Matrix* Cannon(Matrix *matrixA, Matrix *matrixB, int matrixSideDivision);

	static Matrix* algorithmDNS(Matrix *matrixA, Matrix *matrixB, int coreNum);
	static Matrix* DNS(Matrix *matrixA, Matrix *matrixB, int threadCubeDivision);
	static Matrix* matrixMulAndInsertByBlock(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC, int blockIdA, int blockIdB, int sideDivision);
	static int matrixTypeDecision(int typeA, int typeB);
	static Matrix* expandMatrixWithZero(Matrix *outMatrix, int newRow, int newCol);
};
