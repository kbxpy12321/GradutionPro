/********************************************************************************
** Form generated from reading UI file 'QtMatrixOpenMP.ui'
**
** Created by: Qt User Interface Compiler version 5.12.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTMATRIXOPENMP_H
#define UI_QTMATRIXOPENMP_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtMatrixOpenMPClass
{
public:
    QWidget *centralWidget;
    QPushButton *testButton;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QLabel *label_5;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QtMatrixOpenMPClass)
    {
        if (QtMatrixOpenMPClass->objectName().isEmpty())
            QtMatrixOpenMPClass->setObjectName(QString::fromUtf8("QtMatrixOpenMPClass"));
        QtMatrixOpenMPClass->resize(1308, 811);
        centralWidget = new QWidget(QtMatrixOpenMPClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        testButton = new QPushButton(centralWidget);
        testButton->setObjectName(QString::fromUtf8("testButton"));
        testButton->setGeometry(QRect(150, 130, 93, 28));
        label = new QLabel(centralWidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(70, 70, 72, 15));
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(200, 70, 72, 15));
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(370, 70, 72, 15));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setGeometry(QRect(510, 70, 72, 15));
        label_5 = new QLabel(centralWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setGeometry(QRect(410, 130, 72, 15));
        QtMatrixOpenMPClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QtMatrixOpenMPClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1308, 26));
        QtMatrixOpenMPClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QtMatrixOpenMPClass);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        QtMatrixOpenMPClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(QtMatrixOpenMPClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        QtMatrixOpenMPClass->setStatusBar(statusBar);

        retranslateUi(QtMatrixOpenMPClass);

        QMetaObject::connectSlotsByName(QtMatrixOpenMPClass);
    } // setupUi

    void retranslateUi(QMainWindow *QtMatrixOpenMPClass)
    {
        QtMatrixOpenMPClass->setWindowTitle(QApplication::translate("QtMatrixOpenMPClass", "QtMatrixOpenMP", nullptr));
        testButton->setText(QApplication::translate("QtMatrixOpenMPClass", "test", nullptr));
        label->setText(QApplication::translate("QtMatrixOpenMPClass", "TextLabel", nullptr));
        label_2->setText(QApplication::translate("QtMatrixOpenMPClass", "TextLabel", nullptr));
        label_3->setText(QApplication::translate("QtMatrixOpenMPClass", "TextLabel", nullptr));
        label_4->setText(QApplication::translate("QtMatrixOpenMPClass", "TextLabel", nullptr));
        label_5->setText(QApplication::translate("QtMatrixOpenMPClass", "TextLabel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QtMatrixOpenMPClass: public Ui_QtMatrixOpenMPClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTMATRIXOPENMP_H
