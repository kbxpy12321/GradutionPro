#include "QtMatrixOpenMP.h"
#include "Matrix.h"
#include "MatrixCalculation.h"

QtMatrixOpenMP::QtMatrixOpenMP(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.testButton, SIGNAL(clicked()), this, SLOT(clickButton()));
}

#define MATRIXROWA 1679

#define MATRIXSAME 1734

#define MATRIXCOLB 1666


void QtMatrixOpenMP::clickButton() {

}
