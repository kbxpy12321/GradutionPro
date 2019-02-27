#include "QtMatrixOpenMP.h"
#include "Matrix.h"
#include "MatrixCalculation.h"

QtMatrixOpenMP::QtMatrixOpenMP(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.testButton, SIGNAL(clicked()), this, SLOT(clickButton()));
}

#define MATRIXROWA 2047

#define MATRIXSAME 2047

#define MATRIXCOLB 2047


void QtMatrixOpenMP::clickButton() {
	clock_t start, ends;
	
	auto *aa = new Matrix();
	auto *bb = new Matrix();
	aa->randomMatrix(MATRIXROWA, MATRIXSAME, FLOAT, 1, 6);
	bb->randomMatrix(MATRIXSAME, MATRIXCOLB, INTEGER, 1, 6);
	
	start = clock();
	//std::cout << "now start time of normal" << std::endl;
	Matrix *cc = MatrixCalculation::algorithmStrassen(aa, bb, 0, 0);
	ends = clock();
	//std::cout << "now end the time of normal, the time is: " << ends - start << std::endl << std::endl;
	ui.label->setText(QString::number(ends - start));
}
