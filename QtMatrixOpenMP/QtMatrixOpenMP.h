#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtMatrixOpenMP.h"

class QtMatrixOpenMP : public QMainWindow
{
	Q_OBJECT

public:
	QtMatrixOpenMP(QWidget *parent = Q_NULLPTR);

public slots:
	void clickButton();

private:
	Ui::QtMatrixOpenMPClass ui;
};
