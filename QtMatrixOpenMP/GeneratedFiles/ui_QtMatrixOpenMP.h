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
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTableView>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtMatrixOpenMPClass
{
public:
    QWidget *centralWidget;
    QWidget *gridLayoutWidget_2;
    QGridLayout *gridLayout_algochoose;
    QHBoxLayout *horizontalLayout_tableviewcalqin;
    QTableView *tableView_calqueue;
    QHBoxLayout *horizontalLayout_algochoose;
    QComboBox *comboBox_algoParallel;
    QComboBox *comboBox_algoNormal;
    QComboBox *comboBox_corenum;
    QHBoxLayout *horizontalLayout_usetableview;
    QPushButton *pushButton_insertqueque;
    QPushButton *pushButton_deleteque;
    QPushButton *pushButton_startcal;
    QPushButton *pushButton_clear;
    QHBoxLayout *horizontalLayout_name_algochoose;
    QLabel *label_name_algoformer;
    QLabel *label_name_algolatter;
    QLabel *label_name_corenum;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout_rescal;
    QHBoxLayout *horizontalLayout_resbygraph;
    QPushButton *pushButton_coutdata;
    QPushButton *pushButton_showcudares;
    QTableView *tableView_showres;
    QWidget *gridLayoutWidget_5;
    QGridLayout *gridLayout_randommaker;
    QLabel *label_name_matrixmaker;
    QGridLayout *gridLayout_makematrix;
    QLabel *label_name_makematrix;
    QPushButton *pushButton_confirmmake;
    QLabel *label_blank4;
    QLabel *label_blank2;
    QLabel *label_blank3;
    QLabel *label_blank1;
    QGridLayout *gridLayout_matrixbmake;
    QDoubleSpinBox *doubleSpinBox_Bmax;
    QComboBox *comboBox_Btype;
    QDoubleSpinBox *doubleSpinBox_Bmin;
    QLabel *label_name_Bcol;
    QSpinBox *spinBox_Bcol;
    QLabel *label_name_Btype;
    QLabel *label_name_Bmax;
    QLabel *label_name_Bmin;
    QGridLayout *gridLayout_matrixamake;
    QComboBox *comboBox_Atype;
    QLabel *label_name_Atype;
    QLabel *label_name_Amin;
    QSpinBox *spinBox_Arow;
    QLabel *label_name_ARow;
    QLabel *label_name_Amax;
    QDoubleSpinBox *doubleSpinBox_Amin;
    QDoubleSpinBox *doubleSpinBox_Amax;
    QGridLayout *gridLayout_setsameside;
    QLabel *label_name_sameside;
    QSpinBox *spinBox_sameside;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QtMatrixOpenMPClass)
    {
        if (QtMatrixOpenMPClass->objectName().isEmpty())
            QtMatrixOpenMPClass->setObjectName(QString::fromUtf8("QtMatrixOpenMPClass"));
        QtMatrixOpenMPClass->resize(926, 871);
        centralWidget = new QWidget(QtMatrixOpenMPClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayoutWidget_2 = new QWidget(centralWidget);
        gridLayoutWidget_2->setObjectName(QString::fromUtf8("gridLayoutWidget_2"));
        gridLayoutWidget_2->setGeometry(QRect(300, 20, 591, 381));
        gridLayout_algochoose = new QGridLayout(gridLayoutWidget_2);
        gridLayout_algochoose->setSpacing(6);
        gridLayout_algochoose->setContentsMargins(11, 11, 11, 11);
        gridLayout_algochoose->setObjectName(QString::fromUtf8("gridLayout_algochoose"));
        gridLayout_algochoose->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_tableviewcalqin = new QHBoxLayout();
        horizontalLayout_tableviewcalqin->setSpacing(6);
        horizontalLayout_tableviewcalqin->setObjectName(QString::fromUtf8("horizontalLayout_tableviewcalqin"));
        tableView_calqueue = new QTableView(gridLayoutWidget_2);
        tableView_calqueue->setObjectName(QString::fromUtf8("tableView_calqueue"));

        horizontalLayout_tableviewcalqin->addWidget(tableView_calqueue);


        gridLayout_algochoose->addLayout(horizontalLayout_tableviewcalqin, 2, 1, 1, 1);

        horizontalLayout_algochoose = new QHBoxLayout();
        horizontalLayout_algochoose->setSpacing(6);
        horizontalLayout_algochoose->setObjectName(QString::fromUtf8("horizontalLayout_algochoose"));
        comboBox_algoParallel = new QComboBox(gridLayoutWidget_2);
        comboBox_algoParallel->addItem(QString());
        comboBox_algoParallel->addItem(QString());
        comboBox_algoParallel->addItem(QString());
        comboBox_algoParallel->addItem(QString());
        comboBox_algoParallel->setObjectName(QString::fromUtf8("comboBox_algoParallel"));

        horizontalLayout_algochoose->addWidget(comboBox_algoParallel);

        comboBox_algoNormal = new QComboBox(gridLayoutWidget_2);
        comboBox_algoNormal->addItem(QString());
        comboBox_algoNormal->addItem(QString());
        comboBox_algoNormal->addItem(QString());
        comboBox_algoNormal->setObjectName(QString::fromUtf8("comboBox_algoNormal"));

        horizontalLayout_algochoose->addWidget(comboBox_algoNormal);

        comboBox_corenum = new QComboBox(gridLayoutWidget_2);
        comboBox_corenum->setObjectName(QString::fromUtf8("comboBox_corenum"));

        horizontalLayout_algochoose->addWidget(comboBox_corenum);


        gridLayout_algochoose->addLayout(horizontalLayout_algochoose, 1, 1, 1, 1);

        horizontalLayout_usetableview = new QHBoxLayout();
        horizontalLayout_usetableview->setSpacing(6);
        horizontalLayout_usetableview->setObjectName(QString::fromUtf8("horizontalLayout_usetableview"));
        pushButton_insertqueque = new QPushButton(gridLayoutWidget_2);
        pushButton_insertqueque->setObjectName(QString::fromUtf8("pushButton_insertqueque"));

        horizontalLayout_usetableview->addWidget(pushButton_insertqueque);

        pushButton_deleteque = new QPushButton(gridLayoutWidget_2);
        pushButton_deleteque->setObjectName(QString::fromUtf8("pushButton_deleteque"));

        horizontalLayout_usetableview->addWidget(pushButton_deleteque);

        pushButton_startcal = new QPushButton(gridLayoutWidget_2);
        pushButton_startcal->setObjectName(QString::fromUtf8("pushButton_startcal"));

        horizontalLayout_usetableview->addWidget(pushButton_startcal);

        pushButton_clear = new QPushButton(gridLayoutWidget_2);
        pushButton_clear->setObjectName(QString::fromUtf8("pushButton_clear"));

        horizontalLayout_usetableview->addWidget(pushButton_clear);


        gridLayout_algochoose->addLayout(horizontalLayout_usetableview, 3, 1, 1, 1);

        horizontalLayout_name_algochoose = new QHBoxLayout();
        horizontalLayout_name_algochoose->setSpacing(6);
        horizontalLayout_name_algochoose->setObjectName(QString::fromUtf8("horizontalLayout_name_algochoose"));
        label_name_algoformer = new QLabel(gridLayoutWidget_2);
        label_name_algoformer->setObjectName(QString::fromUtf8("label_name_algoformer"));

        horizontalLayout_name_algochoose->addWidget(label_name_algoformer);

        label_name_algolatter = new QLabel(gridLayoutWidget_2);
        label_name_algolatter->setObjectName(QString::fromUtf8("label_name_algolatter"));

        horizontalLayout_name_algochoose->addWidget(label_name_algolatter);

        label_name_corenum = new QLabel(gridLayoutWidget_2);
        label_name_corenum->setObjectName(QString::fromUtf8("label_name_corenum"));

        horizontalLayout_name_algochoose->addWidget(label_name_corenum);


        gridLayout_algochoose->addLayout(horizontalLayout_name_algochoose, 0, 1, 1, 1);

        gridLayoutWidget = new QWidget(centralWidget);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(300, 420, 591, 381));
        gridLayout_rescal = new QGridLayout(gridLayoutWidget);
        gridLayout_rescal->setSpacing(6);
        gridLayout_rescal->setContentsMargins(11, 11, 11, 11);
        gridLayout_rescal->setObjectName(QString::fromUtf8("gridLayout_rescal"));
        gridLayout_rescal->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_resbygraph = new QHBoxLayout();
        horizontalLayout_resbygraph->setSpacing(6);
        horizontalLayout_resbygraph->setObjectName(QString::fromUtf8("horizontalLayout_resbygraph"));
        pushButton_coutdata = new QPushButton(gridLayoutWidget);
        pushButton_coutdata->setObjectName(QString::fromUtf8("pushButton_coutdata"));

        horizontalLayout_resbygraph->addWidget(pushButton_coutdata);

        pushButton_showcudares = new QPushButton(gridLayoutWidget);
        pushButton_showcudares->setObjectName(QString::fromUtf8("pushButton_showcudares"));

        horizontalLayout_resbygraph->addWidget(pushButton_showcudares);


        gridLayout_rescal->addLayout(horizontalLayout_resbygraph, 1, 0, 1, 1);

        tableView_showres = new QTableView(gridLayoutWidget);
        tableView_showres->setObjectName(QString::fromUtf8("tableView_showres"));

        gridLayout_rescal->addWidget(tableView_showres, 0, 0, 1, 1);

        gridLayoutWidget_5 = new QWidget(centralWidget);
        gridLayoutWidget_5->setObjectName(QString::fromUtf8("gridLayoutWidget_5"));
        gridLayoutWidget_5->setGeometry(QRect(10, 30, 261, 681));
        gridLayout_randommaker = new QGridLayout(gridLayoutWidget_5);
        gridLayout_randommaker->setSpacing(6);
        gridLayout_randommaker->setContentsMargins(11, 11, 11, 11);
        gridLayout_randommaker->setObjectName(QString::fromUtf8("gridLayout_randommaker"));
        gridLayout_randommaker->setContentsMargins(0, 0, 0, 0);
        label_name_matrixmaker = new QLabel(gridLayoutWidget_5);
        label_name_matrixmaker->setObjectName(QString::fromUtf8("label_name_matrixmaker"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label_name_matrixmaker->sizePolicy().hasHeightForWidth());
        label_name_matrixmaker->setSizePolicy(sizePolicy);
        QFont font;
        font.setFamily(QString::fromUtf8("Agency FB"));
        font.setPointSize(14);
        label_name_matrixmaker->setFont(font);

        gridLayout_randommaker->addWidget(label_name_matrixmaker, 0, 0, 1, 1);

        gridLayout_makematrix = new QGridLayout();
        gridLayout_makematrix->setSpacing(6);
        gridLayout_makematrix->setObjectName(QString::fromUtf8("gridLayout_makematrix"));
        label_name_makematrix = new QLabel(gridLayoutWidget_5);
        label_name_makematrix->setObjectName(QString::fromUtf8("label_name_makematrix"));

        gridLayout_makematrix->addWidget(label_name_makematrix, 0, 0, 1, 1);

        pushButton_confirmmake = new QPushButton(gridLayoutWidget_5);
        pushButton_confirmmake->setObjectName(QString::fromUtf8("pushButton_confirmmake"));

        gridLayout_makematrix->addWidget(pushButton_confirmmake, 1, 0, 1, 1);


        gridLayout_randommaker->addLayout(gridLayout_makematrix, 10, 0, 1, 1);

        label_blank4 = new QLabel(gridLayoutWidget_5);
        label_blank4->setObjectName(QString::fromUtf8("label_blank4"));

        gridLayout_randommaker->addWidget(label_blank4, 9, 0, 1, 1);

        label_blank2 = new QLabel(gridLayoutWidget_5);
        label_blank2->setObjectName(QString::fromUtf8("label_blank2"));

        gridLayout_randommaker->addWidget(label_blank2, 3, 0, 1, 1);

        label_blank3 = new QLabel(gridLayoutWidget_5);
        label_blank3->setObjectName(QString::fromUtf8("label_blank3"));

        gridLayout_randommaker->addWidget(label_blank3, 7, 0, 1, 1);

        label_blank1 = new QLabel(gridLayoutWidget_5);
        label_blank1->setObjectName(QString::fromUtf8("label_blank1"));

        gridLayout_randommaker->addWidget(label_blank1, 1, 0, 1, 1);

        gridLayout_matrixbmake = new QGridLayout();
        gridLayout_matrixbmake->setSpacing(6);
        gridLayout_matrixbmake->setObjectName(QString::fromUtf8("gridLayout_matrixbmake"));
        doubleSpinBox_Bmax = new QDoubleSpinBox(gridLayoutWidget_5);
        doubleSpinBox_Bmax->setObjectName(QString::fromUtf8("doubleSpinBox_Bmax"));

        gridLayout_matrixbmake->addWidget(doubleSpinBox_Bmax, 3, 1, 1, 1);

        comboBox_Btype = new QComboBox(gridLayoutWidget_5);
        comboBox_Btype->addItem(QString());
        comboBox_Btype->addItem(QString());
        comboBox_Btype->addItem(QString());
        comboBox_Btype->addItem(QString());
        comboBox_Btype->setObjectName(QString::fromUtf8("comboBox_Btype"));
        comboBox_Btype->setEditable(true);

        gridLayout_matrixbmake->addWidget(comboBox_Btype, 1, 1, 1, 1);

        doubleSpinBox_Bmin = new QDoubleSpinBox(gridLayoutWidget_5);
        doubleSpinBox_Bmin->setObjectName(QString::fromUtf8("doubleSpinBox_Bmin"));

        gridLayout_matrixbmake->addWidget(doubleSpinBox_Bmin, 3, 0, 1, 1);

        label_name_Bcol = new QLabel(gridLayoutWidget_5);
        label_name_Bcol->setObjectName(QString::fromUtf8("label_name_Bcol"));

        gridLayout_matrixbmake->addWidget(label_name_Bcol, 0, 0, 1, 1);

        spinBox_Bcol = new QSpinBox(gridLayoutWidget_5);
        spinBox_Bcol->setObjectName(QString::fromUtf8("spinBox_Bcol"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(spinBox_Bcol->sizePolicy().hasHeightForWidth());
        spinBox_Bcol->setSizePolicy(sizePolicy1);
        spinBox_Bcol->setMinimum(1);
        spinBox_Bcol->setMaximum(10000);

        gridLayout_matrixbmake->addWidget(spinBox_Bcol, 1, 0, 1, 1);

        label_name_Btype = new QLabel(gridLayoutWidget_5);
        label_name_Btype->setObjectName(QString::fromUtf8("label_name_Btype"));

        gridLayout_matrixbmake->addWidget(label_name_Btype, 0, 1, 1, 1);

        label_name_Bmax = new QLabel(gridLayoutWidget_5);
        label_name_Bmax->setObjectName(QString::fromUtf8("label_name_Bmax"));

        gridLayout_matrixbmake->addWidget(label_name_Bmax, 2, 1, 1, 1);

        label_name_Bmin = new QLabel(gridLayoutWidget_5);
        label_name_Bmin->setObjectName(QString::fromUtf8("label_name_Bmin"));

        gridLayout_matrixbmake->addWidget(label_name_Bmin, 2, 0, 1, 1);


        gridLayout_randommaker->addLayout(gridLayout_matrixbmake, 8, 0, 1, 1);

        gridLayout_matrixamake = new QGridLayout();
        gridLayout_matrixamake->setSpacing(6);
        gridLayout_matrixamake->setObjectName(QString::fromUtf8("gridLayout_matrixamake"));
        comboBox_Atype = new QComboBox(gridLayoutWidget_5);
        comboBox_Atype->addItem(QString());
        comboBox_Atype->addItem(QString());
        comboBox_Atype->addItem(QString());
        comboBox_Atype->addItem(QString());
        comboBox_Atype->setObjectName(QString::fromUtf8("comboBox_Atype"));
        comboBox_Atype->setEnabled(true);
        comboBox_Atype->setEditable(true);

        gridLayout_matrixamake->addWidget(comboBox_Atype, 1, 1, 1, 1);

        label_name_Atype = new QLabel(gridLayoutWidget_5);
        label_name_Atype->setObjectName(QString::fromUtf8("label_name_Atype"));

        gridLayout_matrixamake->addWidget(label_name_Atype, 0, 1, 1, 1);

        label_name_Amin = new QLabel(gridLayoutWidget_5);
        label_name_Amin->setObjectName(QString::fromUtf8("label_name_Amin"));

        gridLayout_matrixamake->addWidget(label_name_Amin, 2, 0, 1, 1);

        spinBox_Arow = new QSpinBox(gridLayoutWidget_5);
        spinBox_Arow->setObjectName(QString::fromUtf8("spinBox_Arow"));
        spinBox_Arow->setMinimum(1);
        spinBox_Arow->setMaximum(10000);

        gridLayout_matrixamake->addWidget(spinBox_Arow, 1, 0, 1, 1);

        label_name_ARow = new QLabel(gridLayoutWidget_5);
        label_name_ARow->setObjectName(QString::fromUtf8("label_name_ARow"));

        gridLayout_matrixamake->addWidget(label_name_ARow, 0, 0, 1, 1);

        label_name_Amax = new QLabel(gridLayoutWidget_5);
        label_name_Amax->setObjectName(QString::fromUtf8("label_name_Amax"));

        gridLayout_matrixamake->addWidget(label_name_Amax, 2, 1, 1, 1);

        doubleSpinBox_Amin = new QDoubleSpinBox(gridLayoutWidget_5);
        doubleSpinBox_Amin->setObjectName(QString::fromUtf8("doubleSpinBox_Amin"));

        gridLayout_matrixamake->addWidget(doubleSpinBox_Amin, 3, 0, 1, 1);

        doubleSpinBox_Amax = new QDoubleSpinBox(gridLayoutWidget_5);
        doubleSpinBox_Amax->setObjectName(QString::fromUtf8("doubleSpinBox_Amax"));

        gridLayout_matrixamake->addWidget(doubleSpinBox_Amax, 3, 1, 1, 1);


        gridLayout_randommaker->addLayout(gridLayout_matrixamake, 2, 0, 1, 1);

        gridLayout_setsameside = new QGridLayout();
        gridLayout_setsameside->setSpacing(6);
        gridLayout_setsameside->setObjectName(QString::fromUtf8("gridLayout_setsameside"));
        label_name_sameside = new QLabel(gridLayoutWidget_5);
        label_name_sameside->setObjectName(QString::fromUtf8("label_name_sameside"));

        gridLayout_setsameside->addWidget(label_name_sameside, 0, 0, 1, 1);

        spinBox_sameside = new QSpinBox(gridLayoutWidget_5);
        spinBox_sameside->setObjectName(QString::fromUtf8("spinBox_sameside"));
        spinBox_sameside->setMinimum(1);
        spinBox_sameside->setMaximum(10000);

        gridLayout_setsameside->addWidget(spinBox_sameside, 1, 0, 1, 1);


        gridLayout_randommaker->addLayout(gridLayout_setsameside, 4, 0, 1, 1);

        QtMatrixOpenMPClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QtMatrixOpenMPClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 926, 26));
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
        comboBox_algoParallel->setItemText(0, QApplication::translate("QtMatrixOpenMPClass", "\346\227\240", nullptr));
        comboBox_algoParallel->setItemText(1, QApplication::translate("QtMatrixOpenMPClass", "Stassen\345\271\266\350\241\214\345\214\226\347\256\227\346\263\225", nullptr));
        comboBox_algoParallel->setItemText(2, QApplication::translate("QtMatrixOpenMPClass", "Cannon\345\271\266\350\241\214\347\256\227\346\263\225", nullptr));
        comboBox_algoParallel->setItemText(3, QApplication::translate("QtMatrixOpenMPClass", "DNS\345\271\266\350\241\214\347\256\227\346\263\225", nullptr));

        comboBox_algoNormal->setItemText(0, QApplication::translate("QtMatrixOpenMPClass", "\344\274\240\347\273\237\347\237\251\351\230\265\344\271\230\346\263\225", nullptr));
        comboBox_algoNormal->setItemText(1, QApplication::translate("QtMatrixOpenMPClass", "\344\274\240\347\273\237\345\271\266\350\241\214\347\237\251\351\230\265\344\271\230\346\263\225", nullptr));
        comboBox_algoNormal->setItemText(2, QApplication::translate("QtMatrixOpenMPClass", "Strassen\347\256\227\346\263\225", nullptr));

        pushButton_insertqueque->setText(QApplication::translate("QtMatrixOpenMPClass", "\346\217\222\345\205\245\350\256\241\347\256\227\351\230\237\345\210\227", nullptr));
        pushButton_deleteque->setText(QApplication::translate("QtMatrixOpenMPClass", "\344\273\216\350\256\241\347\256\227\351\230\237\345\210\227\345\210\240\351\231\244", nullptr));
        pushButton_startcal->setText(QApplication::translate("QtMatrixOpenMPClass", "\345\274\200\345\247\213\350\256\241\347\256\227", nullptr));
        pushButton_clear->setText(QApplication::translate("QtMatrixOpenMPClass", "\345\205\250\351\203\250\346\270\205\347\251\272", nullptr));
        label_name_algoformer->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265\345\210\222\345\210\206\347\256\227\346\263\225\351\200\211\346\213\251", nullptr));
        label_name_algolatter->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265\347\233\270\344\271\230\347\256\227\346\263\225\351\200\211\346\213\251", nullptr));
        label_name_corenum->setText(QApplication::translate("QtMatrixOpenMPClass", "cpu\346\240\270\345\277\203\346\225\260", nullptr));
        pushButton_coutdata->setText(QApplication::translate("QtMatrixOpenMPClass", "\345\257\274\345\207\272\346\225\260\346\215\256", nullptr));
        pushButton_showcudares->setText(QApplication::translate("QtMatrixOpenMPClass", "CudaMul", nullptr));
        label_name_matrixmaker->setText(QApplication::translate("QtMatrixOpenMPClass", "\351\232\217\346\234\272\347\237\251\351\230\265\347\224\237\346\210\220\345\231\250", nullptr));
        label_name_makematrix->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\224\237\346\210\220\347\237\251\351\230\265", nullptr));
        pushButton_confirmmake->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\241\256\350\256\244\347\224\237\346\210\220", nullptr));
        label_blank4->setText(QString());
        label_blank2->setText(QString());
        label_blank3->setText(QString());
        label_blank1->setText(QString());
        comboBox_Btype->setItemText(0, QApplication::translate("QtMatrixOpenMPClass", "Integer", nullptr));
        comboBox_Btype->setItemText(1, QApplication::translate("QtMatrixOpenMPClass", "Float", nullptr));
        comboBox_Btype->setItemText(2, QApplication::translate("QtMatrixOpenMPClass", "Double", nullptr));
        comboBox_Btype->setItemText(3, QApplication::translate("QtMatrixOpenMPClass", "Long Long", nullptr));

        label_name_Bcol->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\345\210\227\346\225\260", nullptr));
        label_name_Btype->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\347\261\273\345\236\213", nullptr));
        label_name_Bmax->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\346\234\200\345\244\247\345\200\274", nullptr));
        label_name_Bmin->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\346\234\200\345\260\217\345\200\274", nullptr));
        comboBox_Atype->setItemText(0, QApplication::translate("QtMatrixOpenMPClass", "Integer", nullptr));
        comboBox_Atype->setItemText(1, QApplication::translate("QtMatrixOpenMPClass", "Float", nullptr));
        comboBox_Atype->setItemText(2, QApplication::translate("QtMatrixOpenMPClass", "Double", nullptr));
        comboBox_Atype->setItemText(3, QApplication::translate("QtMatrixOpenMPClass", "Long Long", nullptr));

        label_name_Atype->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\347\261\273\345\236\213", nullptr));
        label_name_Amin->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\346\234\200\345\260\217\345\200\274", nullptr));
        label_name_ARow->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\350\241\214\346\225\260", nullptr));
        label_name_Amax->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\346\234\200\345\244\247\345\200\274", nullptr));
        label_name_sameside->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\233\270\345\220\214\350\276\271\351\225\277", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QtMatrixOpenMPClass: public Ui_QtMatrixOpenMPClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTMATRIXOPENMP_H
