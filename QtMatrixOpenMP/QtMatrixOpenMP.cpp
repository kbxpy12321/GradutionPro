#include "QtMatrixOpenMP.h"
#include "Matrix.h"
#include "MatrixCalculation.h"

#include <QStandardItemModel>
#include <QMessageBox>
#include <omp.h>

#include <CudaMatrixCalculation.cuh>

#define tableViewCalQueueCol 4
#define tableViewShowRes 3

QStandardItemModel *modelInTableViewCalQueue;
QStandardItemModel *modelInTableViewShowRes;
Matrix *matrixA, *matrixB;
int algoArray = 0;

extern "C" Matrix *matrixMulByCuda(Matrix *matrixA, Matrix *matrixB);
extern "C" int cudaGetGpus();

void QtMatrixOpenMP::initButton() {
	connect(ui.pushButton_insertqueque, SIGNAL(clicked()), this, SLOT(clickPushButton_InsertQueque()));
	connect(ui.pushButton_deleteque, SIGNAL(clicked()), this, SLOT(clickPushButton_DeleteQueue()));
	connect(ui.pushButton_confirmmake, SIGNAL(clicked()), this, SLOT(clickPushButton_ConformMake()));
	connect(ui.pushButton_clear, SIGNAL(clicked()), this, SLOT(clickPushButton_ClearBox()));
	connect(ui.pushButton_startcal, SIGNAL(clicked()), this, SLOT(clickPushButton_StartCalculation()));
	connect(ui.pushButton_showcudares, SIGNAL(clicked()), this, SLOT(clickPushButton_ShowCudaRes()));
}

void QtMatrixOpenMP::initTableView() {
	modelInTableViewCalQueue = new QStandardItemModel();
	modelInTableViewCalQueue->setColumnCount(tableViewCalQueueCol);
	modelInTableViewCalQueue->setHeaderData(0, Qt::Horizontal, QString::fromLocal8Bit("矩阵划分算法"));
	modelInTableViewCalQueue->setHeaderData(1, Qt::Horizontal, QString::fromLocal8Bit("矩阵相乘算法"));
	modelInTableViewCalQueue->setHeaderData(2, Qt::Horizontal, QString::fromLocal8Bit("CPU核心数"));
	modelInTableViewCalQueue->setHeaderData(3, Qt::Horizontal, QString::fromLocal8Bit("GPU数"));
	ui.tableView_calqueue->setModel(modelInTableViewCalQueue);
	ui.tableView_calqueue->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.tableView_calqueue->setSelectionBehavior(QAbstractItemView::SelectRows);


	modelInTableViewShowRes = new QStandardItemModel();
	modelInTableViewShowRes->setColumnCount(tableViewShowRes);
	modelInTableViewShowRes->setHeaderData(0, Qt::Horizontal, QString::fromLocal8Bit("矩阵划分算法"));
	modelInTableViewShowRes->setHeaderData(1, Qt::Horizontal, QString::fromLocal8Bit("矩阵相乘算法"));
	modelInTableViewShowRes->setHeaderData(2, Qt::Horizontal, QString::fromLocal8Bit("计算时间"));
	ui.tableView_showres->setModel(modelInTableViewShowRes);
	ui.tableView_showres->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.tableView_showres->setSelectionBehavior(QAbstractItemView::SelectRows);
}

void QtMatrixOpenMP::initCombobox() {
	int totalCoreNum = omp_get_num_procs();
	for (int i = 1; i <= totalCoreNum; i++) {
		ui.comboBox_corenum->addItem(QString::number(i));
	}
	int totalCudaGpus = cudaGetGpus();
	for (int i = 0; i <= totalCudaGpus; i++) {
		ui.comboBox_gpunum->addItem(QString::number(i));
	}
}

QtMatrixOpenMP::QtMatrixOpenMP(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	initButton();
	initTableView();
	initCombobox();
}

void QtMatrixOpenMP::clickPushButton_InsertQueque() {
	QList<QStandardItem*> itemIntoTableViewCalQueue;
	itemIntoTableViewCalQueue.append(new QStandardItem(ui.comboBox_algoParallel->currentText()));
	itemIntoTableViewCalQueue.append(new QStandardItem(ui.comboBox_algoNormal->currentText()));
	itemIntoTableViewCalQueue.append(new QStandardItem(ui.comboBox_corenum->currentText()));
	itemIntoTableViewCalQueue.append(new QStandardItem(ui.comboBox_gpunum->currentText()));
	modelInTableViewCalQueue->appendRow(itemIntoTableViewCalQueue);
}

void QtMatrixOpenMP::clickPushButton_DeleteQueue() {
	//ui.pushButton_clear->setText(QString::number(ui.tableView_calqueue->currentIndex().row()));
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
	matrixA = new Matrix();
	matrixB = new Matrix();
	matrixA->randomMatrix(row, sameSide, typeA, minA, maxA);
	matrixB->randomMatrix(sameSide, col, typeB, minB, maxB);

	std::string messageToAlert;

	if (matrixA != NULL && matrixB != NULL) {
		matrixA->writeMatrix("matrixA.txt");
		matrixB->writeMatrix("matrixB.txt");
		messageToAlert += "matrixA row: " + std::to_string(matrixA->getRow()) + " col: " + std::to_string(matrixA->getCol()) + " type: " + std::to_string(matrixA->getType());
		messageToAlert += "\nmatrixB row: " + std::to_string(matrixB->getRow()) + " col: " + std::to_string(matrixB->getCol()) + " type: " + std::to_string(matrixB->getType());
		QMessageBox::about(NULL, "success!",QString::fromStdString(messageToAlert));
	}
	else
	{
		QMessageBox::warning(NULL, "", "failed", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
	}
}

void QtMatrixOpenMP::clickPushButton_ClearBox() {
	modelInTableViewCalQueue->removeRows(0, modelInTableViewCalQueue->rowCount());
}
void QtMatrixOpenMP::clickPushButton_StartCalculation() {
	if (matrixA == NULL || matrixB == NULL) {
		QMessageBox::warning(NULL, "", "No matrix!", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
		return;
	}
	modelInTableViewShowRes->removeRows(0, modelInTableViewShowRes->rowCount());
	int tmpRow = modelInTableViewCalQueue->rowCount();
	int tmpCol = tableViewCalQueueCol;
	for (int i = 0; i < tmpRow; i++) {
		QString algoFormer = modelInTableViewCalQueue->data(modelInTableViewCalQueue->index(i, 0)).toString();
		QString algoLatter = modelInTableViewCalQueue->data(modelInTableViewCalQueue->index(i, 1)).toString();
		int coreNum = modelInTableViewCalQueue->data(modelInTableViewCalQueue->index(i, 2)).toInt();
		int gpuNum = modelInTableViewCalQueue->data(modelInTableViewCalQueue->index(i, 3)).toInt();
		algoArray = i + 1;
		doAlgo(algoFormer, algoLatter, coreNum, gpuNum);
	}
	//ui.pushButton_clear->setText(QString::number(modelInTableViewCalQueue->rowCount()));
}
int algoNameToCode(QString algo) {
	if (algo == QString::fromLocal8Bit("无")) {
		return NOALGO;
	}
	if (algo == QString::fromLocal8Bit("Stassen并行化算法")) {
		return ALGOSTRASSENPARALLEL;
	}
	if (algo == QString::fromLocal8Bit("Cannon并行算法")) {
		return ALGOCANNON;
	}
	if (algo == QString::fromLocal8Bit("DNS并行算法")) {
		return ALGODNS;
	}
	if (algo == QString::fromLocal8Bit("朴素矩阵乘法")) {
		return NORMALMATRIXMUL;
	}
	if (algo == QString::fromLocal8Bit("朴素并行乘法")) {
		return NORMALMATRIXMULPARALLEL;
	}
	if (algo == QString::fromLocal8Bit("Strassen算法")) {
		return ALGOSTRASSEN;
	}
	return ERRORCODE;
}

void QtMatrixOpenMP::doAlgo(QString algoFormer, QString algoLatter, int coreNum, int gpuNum) {
	int algoFormerCode = algoNameToCode(algoFormer);
	int algoLatterCode = algoNameToCode(algoLatter);
	Matrix *matrixRes;
	clock_t start, end;
	start = clock();
	if (gpuNum != 0) {
		matrixRes = matrixMulByCuda(matrixA, matrixB);
	}
	else {
		if (algoFormerCode == ALGODNS) {
			matrixRes = MatrixCalculation::algorithmDNS(matrixA, matrixB, coreNum, algoLatterCode);
		}
		else if (algoFormerCode == ALGOCANNON) {
			matrixRes = MatrixCalculation::algorithmCannon(matrixA, matrixB, coreNum, algoLatterCode);
		}
		else if (algoFormer == ALGOSTRASSENPARALLEL) {
			matrixRes = MatrixCalculation::algorithmStrassen(matrixA, matrixB, coreNum, algoLatterCode);
		}
		else {
			if (algoLatterCode == ALGOSTRASSEN) {
				matrixRes = MatrixCalculation::algorithmStrassen(matrixA, matrixB, 0, 0);
			}
			else if (algoLatterCode == NORMALMATRIXMULPARALLEL) {
				matrixRes = MatrixCalculation::matrixMulParallel(matrixA, matrixB, coreNum);
			}
			else {
				matrixRes = MatrixCalculation::matrixMul(matrixA, matrixB);
			}
		}
	}
	end = clock();
	QList<QStandardItem*> itemIntoTableViewShowRes;
	itemIntoTableViewShowRes.append(new QStandardItem(algoFormer));
	itemIntoTableViewShowRes.append(new QStandardItem(algoLatter));
	itemIntoTableViewShowRes.append(new QStandardItem(QString::number(end - start)));
	modelInTableViewShowRes->appendRow(itemIntoTableViewShowRes);
	std::string resFileName = "res" + std::to_string(algoArray) + ".txt";
	matrixRes->writeMatrix(resFileName);
}

void QtMatrixOpenMP::clickPushButton_ShowCudaRes()
{
	//clock_t start, end;
	//start = clock();
	////ui.pushButton_showcudares->setText(QString::number(testCuda::testInCuda()));
	//Matrix *test1 = matrixMulByCuda(matrixA, matrixB);//TODO
	////Matrix *test2 = MatrixCalculation::matrixMul(matrixA, matrixB);
	//end = clock();
	//test1->printMatrix();
	//Matrix *test2 = MatrixCalculation::matrixMul(matrixA, matrixB);
	//int a = test2->matrixCompare(test1);
	//ui.pushButton_showcudares->setText(QString::number(a));
	

	//QList<QStandardItem*> itemIntoTableViewShowRes;
	//itemIntoTableViewShowRes.append(new QStandardItem(QString::fromLocal8Bit("Cuda并行算法")));
	//itemIntoTableViewShowRes.append(new QStandardItem(QString::fromLocal8Bit("无")));
	//itemIntoTableViewShowRes.append(new QStandardItem(QString::number(end - start)));
	//modelInTableViewShowRes->appendRow(itemIntoTableViewShowRes);
}

