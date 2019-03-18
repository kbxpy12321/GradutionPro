//
// Created by LZR on 2019/1/4.
//

#include "Matrix.h"
#include "MatrixCalculation.h"
#include <stdlib.h>
#include <iostream>
#include <random>
#include <vector>

void Matrix::randomMatrix(int row, int col, int matrixType, int MIN, int MAX) {
	std::random_device rd;
	std::default_random_engine e1(rd());
	std::uniform_int_distribution<int> uniform_dist(MIN, MAX);
	this->matrixType = matrixType;
	setRow(row);
	setCol(col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			this->matrixPush(uniform_dist(e1));
		}
	}
}

void Matrix::readMatrix() {

}

std::vector<int> Matrix::returnVector() {
	return integerMatrix;
}

void Matrix::printMatrix() {
	int row = getRow();
	int col = getCol();
	for (int i = 1; i <= row; i++) {
		for (int j = 1; j <= col; j++) {
			std::cout << getMatrixElement(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int Matrix::getRow() {
	return row;
}

int Matrix::getCol() {
	return col;
}

Matrix* Matrix::normalMatrixMultiple(Matrix* outMatrix) {
	int outRow = outMatrix->getRow();
	int col = getCol();
	if (outRow != col) {
		std::cout << "Wrong Matrix Multiple" << std::endl;
		return this;
	}
	int row = getRow();
	int outCol = outMatrix->getCol();
	setCol(outCol);
	std::vector<int> tmpMatrix;
	double sumPerLine = 0;
	for (int i = 1; i <= row; i++) {
		for (int k = 1; k <= outCol; k++) {
			for (int j = 1; j <= col; j++) {
				sumPerLine += getMatrixElement(i, j) * outMatrix->getMatrixElement(j, k);
			}
			tmpMatrix.push_back(sumPerLine);
			sumPerLine = 0;
		}
	}
	integerMatrix.clear();
	integerMatrix.shrink_to_fit();
	integerMatrix = tmpMatrix;
	return this;
}

Matrix::Matrix(int Type) {
	matrixType = Type;
	setRow(0);
	setCol(0);
}

Matrix::Matrix() {
	matrixType = INTEGER;
	setRow(0);
	setCol(0);
}

void Matrix::setCol(int col) {
	this->col = col;
}

void Matrix::setRow(int row) {
	this->row = row;
}

double Matrix::getMatrixElement(int x, int y) {
	if (matrixType == INTEGER) {
		return integerMatrix[(x - 1) * col + y - 1];
	}
	if (matrixType == FLOAT) {
		return floatMatrix[(x - 1) * col + y - 1];
	}
	if (matrixType == DOUBLE) {
		return doubleMatrix[(x - 1) * col + y - 1];
	}
	if (matrixType == LONGLONG) {
		return longMatrix[(x - 1) * col + y - 1];
	}
	return ERRORCODE;
}


double Matrix::getMatrixElement(int i) {
	if (matrixType == INTEGER) {
		return integerMatrix[i];
	}
	else if (matrixType == FLOAT) {
		return floatMatrix[i];
	}
	else if (matrixType == DOUBLE) {
		return doubleMatrix[i];
	}
	else if (matrixType == LONGLONG) {
		return longMatrix[i];
	}
	else
		return ERRORCODE;
}

void Matrix::setMatrixElement(int x, int y, double val) {
	if (x > row || y > col)
		return;
	if (matrixType == INTEGER) {
		integerMatrix[(x - 1) * col + y - 1] = (int)val;
	}
	else if (matrixType == FLOAT) {
		floatMatrix[(x - 1) * col + y - 1] = (float)val;
	}
	else if (matrixType == DOUBLE) {
		doubleMatrix[(x - 1) * col + y - 1] = val;
	}
	else if (matrixType == LONGLONG) {
		longMatrix[(x - 1) * col + y - 1] = (long long)val;
	}
}

void Matrix::setMatrixElement(int i, double val) {
	if (matrixType == INTEGER) {
		integerMatrix[i] = (int)val;
	}
	else if (matrixType == FLOAT) {
		floatMatrix[i] = (float)val;
	}
	else if (matrixType == DOUBLE) {
		doubleMatrix[i] = val;
	}
	else if (matrixType == LONGLONG) {
		longMatrix[i] = (long long)val;
	}
}


void Matrix::matrixAdd(Matrix *outMatrix) {
	if (row != outMatrix->row || col != outMatrix->col) {
		return;
	}
	int tmpLen = row * col;
	if (matrixType == INTEGER) {
		for (int i = 0; i < tmpLen; i++) {
			integerMatrix[i] += (int)outMatrix->getMatrixElement(i);
		}
	}
	else if (matrixType == FLOAT) {
		for (int i = 0; i < tmpLen; i++) {
			floatMatrix[i] += (float)outMatrix->getMatrixElement(i);
		}
	}
	else if (matrixType == DOUBLE) {
		for (int i = 0; i < tmpLen; i++) {
			doubleMatrix[i] += outMatrix->getMatrixElement(i);
		}
	}
	else if (matrixType == LONGLONG) {
		for (int i = 0; i < tmpLen; i++) {
			longMatrix[i] += (long long)outMatrix->getMatrixElement(i);
		}
	}
}

void Matrix::matrixSub(Matrix *outMatrix) {
	if (row != outMatrix->row || col != outMatrix->col) {
		return;
	}
	int tmpLen = row * col;
	if (matrixType == INTEGER) {
		for (int i = 0; i < tmpLen; i++) {
			integerMatrix[i] -= (int)outMatrix->getMatrixElement(i);
		}
	}
	else if (matrixType == FLOAT) {
		for (int i = 0; i < tmpLen; i++) {
			floatMatrix[i] -= (float)outMatrix->getMatrixElement(i);
		}
	}
	else if (matrixType == DOUBLE) {
		for (int i = 0; i < tmpLen; i++) {
			doubleMatrix[i] -= outMatrix->getMatrixElement(i);
		}
	}
	else if (matrixType == LONGLONG) {
		for (int i = 0; i < tmpLen; i++) {
			longMatrix[i] -= (long long)outMatrix->getMatrixElement(i);
		}
	}
}



Matrix *Matrix::generateMatrixParts(int leftTopRow, int leftTopCol, int rightDownRow, int rightDownCol) {
	if (leftTopRow < 1 || leftTopCol < 1 || rightDownRow > getRow() || rightDownCol > getCol())
		return this;

	if ((getType() == INTEGER && integerMatrix.empty()) || (getType() == FLOAT && floatMatrix.empty())
		|| (getType() == DOUBLE && doubleMatrix.empty()) || (getType() == LONGLONG && longMatrix.empty()))
		return this;

	auto *tmpMatrix = new Matrix(rightDownRow - leftTopRow + 1, rightDownCol - leftTopCol + 1, this->getType());
	for (int i = leftTopRow; i <= rightDownRow; i++) {
		for (int j = leftTopCol; j <= rightDownCol; j++) {
			tmpMatrix->matrixPush(getMatrixElement(i, j));
		}
	}
	return tmpMatrix;
}

void Matrix::matrixPush(double x) {
	if (matrixType == INTEGER) {
		integerMatrix.push_back((int)x);
	}
	else if (matrixType == FLOAT) {
		floatMatrix.push_back((float)x);
	}
	else if (matrixType == DOUBLE) {
		doubleMatrix.push_back(x);
	}
	else if (matrixType == LONGLONG) {
		longMatrix.push_back((long long)x);
	}
}

Matrix::Matrix(int row, int col, int type) {
	setRow(row);
	setCol(col);
	matrixType = type;
}

void Matrix::initVectorSpace() {
	int fullLen = row * col;
	if (matrixType == INTEGER) {
		integerMatrix.resize((unsigned)fullLen, 0);
	}
	else if (matrixType == FLOAT) {
		floatMatrix.resize((unsigned)fullLen, 0);
	}
	else if (matrixType == DOUBLE) {
		doubleMatrix.resize((unsigned)fullLen, 0);
	}
	else if (matrixType == LONGLONG) {
		longMatrix.resize((unsigned)fullLen, 0);
	}
}

void Matrix::addIntoMatrixElement(int x, int y, double val) {
	if (x > row || y > col)
		return;
	int type = this->getType();
	if (type == INTEGER) {
		integerMatrix[(x - 1) * col + y - 1] += (int)val;
	}
	else if (type == FLOAT) {
		floatMatrix[(x - 1) * col + y - 1] += (float)val;
	}
	else if (type == DOUBLE) {
		doubleMatrix[(x - 1) * col + y - 1] += val;
	}
	else if (type == LONGLONG) {
		longMatrix[(x - 1) * col + y - 1] += (long long)val;
	}
}

bool Matrix::matrixCompare(Matrix *outMatrix) {
	if (this->getType() != outMatrix->getType())
		return false;
	else if (this->getRow() != outMatrix->getRow() || this->getCol() != outMatrix->getCol())
		return false;
	else {
		int tmpLen = row * col;
		for (int i = 0; i < tmpLen; i++) {
			if (this->getMatrixElement(i) != outMatrix->getMatrixElement(i))
				return false;
		}
	}
	return true;
}

int Matrix::getType() {
	return matrixType;
}

void Matrix::changeType(int type) {
	if (type != LONGLONG && type != DOUBLE && type != INTEGER && type != FLOAT)
		return;
	clearTypeMatrix();
	this->matrixType = type;
}

Matrix::~Matrix() {
	clearTypeMatrix();
	//std::cout<<"all memory have out"<<std::endl;
}

void Matrix::clearTypeMatrix() {
	int type = matrixType;
	if (type == INTEGER) {
		integerMatrix.clear();
		integerMatrix.shrink_to_fit();
	}
	else if (type == FLOAT) {
		floatMatrix.clear();
		floatMatrix.shrink_to_fit();
	}
	else if (type == DOUBLE) {
		doubleMatrix.clear();
		doubleMatrix.shrink_to_fit();
	}
	else if (type == LONGLONG) {
		longMatrix.clear();
		longMatrix.shrink_to_fit();
	}
	else {
		return;
	}
	setRow(0);
	setCol(0);
}
