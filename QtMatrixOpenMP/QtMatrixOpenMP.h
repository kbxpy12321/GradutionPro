#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtMatrixOpenMP.h"

class QtMatrixOpenMP : public QMainWindow
{
	Q_OBJECT

public:
	QtMatrixOpenMP(QWidget *parent = Q_NULLPTR);

public slots:
	void clickPushButton_InsertQueque();
	void clickPushButton_DeleteQueue();
	void clickPushButton_StartCalculation();
	void clickPushButton_ClearBox();
	void clickPushButton_ConformMake();
	void clickPushButton_ShowCudaRes();

private:
	Ui::QtMatrixOpenMPClass ui;
	void doAlgo(QString algoFormer, QString AlgoLatter, int coreNum);
	void initButton();
	void initTableView();
	void initCombobox();
};
