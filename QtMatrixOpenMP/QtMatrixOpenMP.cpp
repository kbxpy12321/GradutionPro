#include "QtMatrixOpenMP.h"
#include "Matrix.h"
#include "MatrixCalculation.h"

#include <QStandardItemModel>
#include <omp.h>

QStandardItemModel *modelInTableViewCalQueue;
Matrix *matrixA, *matrixB;

QtMatrixOpenMP::QtMatrixOpenMP(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	
	connect(ui.pushButton_insertqueque, SIGNAL(clicked()), this, SLOT(clickPushButton_InsertQueque()));
	connect(ui.pushButton_deleteque, SIGNAL(clicked()), this, SLOT(clickPushButton_DeleteQueue()));
	connect(ui.pushButton_confirmmake, SIGNAL(clicked()), this, SLOT(clickPushButton_ConformMake()));
	
	int totalCoreNum = omp_get_num_procs();
	for (int i = 1; i <= totalCoreNum; i++) {
		ui.comboBox_corenum->addItem(QString::number(i));
	}
	modelInTableViewCalQueue = new QStandardItemModel();
	modelInTableViewCalQueue->setColumnCount(3);
	modelInTableViewCalQueue->setHeaderData(0, Qt::Horizontal, QString::fromLocal8Bit("前置算法"));
	modelInTableViewCalQueue->setHeaderData(1, Qt::Horizontal, QString::fromLocal8Bit("后置算法"));
	modelInTableViewCalQueue->setHeaderData(2, Qt::Horizontal, QString::fromLocal8Bit("核心数"));
	ui.tableView_calqueue->setModel(modelInTableViewCalQueue);
	ui.tableView_calqueue->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}

void QtMatrixOpenMP::clickPushButton_InsertQueque() {
	QList<QStandardItem*> itemIntoTableViewCalQueue;
	itemIntoTableViewCalQueue.append(new QStandardItem(ui.comboBox_algoParallel->currentText()));
	itemIntoTableViewCalQueue.append(new QStandardItem(ui.comboBox_algoNormal->currentText()));
	itemIntoTableViewCalQueue.append(new QStandardItem(ui.comboBox_corenum->currentText()));
	modelInTableViewCalQueue->appendRow(itemIntoTableViewCalQueue);
}

void QtMatrixOpenMP::clickPushButton_DeleteQueue() {
	modelInTableViewCalQueue->removeRow(ui.tableView_calqueue->currentIndex().row());
}

void QtMatrixOpenMP::clickPushButton_ConformMake() {
	if (matrixA != NULL) {
		delete matrixA;
	}
	if (matrixB != NULL) {
		delete matrixB;
	}
	int row, col, sameSide, typeA, typeB;
	double minA, maxA, minB, maxB;
	row = ui.spinBox_Arow->value();
	sameSide = ui.spinBox_sameside->value();
	col = ui.spinBox_Bcol->value();
	typeA = ui.comboBox_Atype->currentIndex();
	typeB = ui.comboBox_Btype->currentIndex();
	minA = ui.doubleSpinBox_Amin->value();
	maxA = ui.doubleSpinBox_Amax->value();
	minB = ui.doubleSpinBox_Bmin->value();
	maxB = ui.doubleSpinBox_Bmax->value();
	matrixA->randomMatrix(row, sameSide, typeA, minA, maxA);
	matrixB->randomMatrix(sameSide, col, typeB, minB, maxB);
}


