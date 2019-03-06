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
	//void clickPushButton_StartCalculation();
	//void clickPushButton_ClearBox();
	void clickPushButton_ConformMake();

private:
	Ui::QtMatrixOpenMPClass ui;
};
