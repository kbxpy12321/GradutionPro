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
    QWidget *gridLayoutWidget_3;
    QGridLayout *gridLayout_randommatrix;
    QHBoxLayout *horizontalLayout_name_matrixpara;
    QLabel *label_name_ARow;
    QLabel *label_name_sameside;
    QLabel *label_name_Bcol;
    QLabel *label_name_Atype;
    QLabel *label_name_Btype;
    QLabel *label_name_Amin;
    QLabel *label_name_Amax;
    QLabel *label_name_Bmin;
    QLabel *label_name_Bmax;
    QLabel *label_name_makematrix;
    QHBoxLayout *horizontalLayout_name_randommatrix;
    QLabel *label_name_matrixmaker;
    QHBoxLayout *horizontalLayout_matrixpara;
    QSpinBox *spinBox_Arow;
    QSpinBox *spinBox_sameside;
    QSpinBox *spinBox_Bcol;
    QComboBox *comboBox_Atype;
    QComboBox *comboBox_Btype;
    QDoubleSpinBox *doubleSpinBox_Amin;
    QDoubleSpinBox *doubleSpinBox_Amax;
    QDoubleSpinBox *doubleSpinBox_Bmin;
    QDoubleSpinBox *doubleSpinBox_Bmax;
    QPushButton *pushButton_confirmmake;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout_rescal;
    QTableView *tableView_showres;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QtMatrixOpenMPClass)
    {
        if (QtMatrixOpenMPClass->objectName().isEmpty())
            QtMatrixOpenMPClass->setObjectName(QString::fromUtf8("QtMatrixOpenMPClass"));
        QtMatrixOpenMPClass->resize(1221, 564);
        centralWidget = new QWidget(QtMatrixOpenMPClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayoutWidget_2 = new QWidget(centralWidget);
        gridLayoutWidget_2->setObjectName(QString::fromUtf8("gridLayoutWidget_2"));
        gridLayoutWidget_2->setGeometry(QRect(10, 110, 591, 381));
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

        gridLayoutWidget_3 = new QWidget(centralWidget);
        gridLayoutWidget_3->setObjectName(QString::fromUtf8("gridLayoutWidget_3"));
        gridLayoutWidget_3->setGeometry(QRect(10, 10, 1191, 93));
        gridLayout_randommatrix = new QGridLayout(gridLayoutWidget_3);
        gridLayout_randommatrix->setSpacing(6);
        gridLayout_randommatrix->setContentsMargins(11, 11, 11, 11);
        gridLayout_randommatrix->setObjectName(QString::fromUtf8("gridLayout_randommatrix"));
        gridLayout_randommatrix->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_name_matrixpara = new QHBoxLayout();
        horizontalLayout_name_matrixpara->setSpacing(6);
        horizontalLayout_name_matrixpara->setObjectName(QString::fromUtf8("horizontalLayout_name_matrixpara"));
        label_name_ARow = new QLabel(gridLayoutWidget_3);
        label_name_ARow->setObjectName(QString::fromUtf8("label_name_ARow"));

        horizontalLayout_name_matrixpara->addWidget(label_name_ARow);

        label_name_sameside = new QLabel(gridLayoutWidget_3);
        label_name_sameside->setObjectName(QString::fromUtf8("label_name_sameside"));

        horizontalLayout_name_matrixpara->addWidget(label_name_sameside);

        label_name_Bcol = new QLabel(gridLayoutWidget_3);
        label_name_Bcol->setObjectName(QString::fromUtf8("label_name_Bcol"));

        horizontalLayout_name_matrixpara->addWidget(label_name_Bcol);

        label_name_Atype = new QLabel(gridLayoutWidget_3);
        label_name_Atype->setObjectName(QString::fromUtf8("label_name_Atype"));

        horizontalLayout_name_matrixpara->addWidget(label_name_Atype);

        label_name_Btype = new QLabel(gridLayoutWidget_3);
        label_name_Btype->setObjectName(QString::fromUtf8("label_name_Btype"));

        horizontalLayout_name_matrixpara->addWidget(label_name_Btype);

        label_name_Amin = new QLabel(gridLayoutWidget_3);
        label_name_Amin->setObjectName(QString::fromUtf8("label_name_Amin"));

        horizontalLayout_name_matrixpara->addWidget(label_name_Amin);

        label_name_Amax = new QLabel(gridLayoutWidget_3);
        label_name_Amax->setObjectName(QString::fromUtf8("label_name_Amax"));

        horizontalLayout_name_matrixpara->addWidget(label_name_Amax);

        label_name_Bmin = new QLabel(gridLayoutWidget_3);
        label_name_Bmin->setObjectName(QString::fromUtf8("label_name_Bmin"));

        horizontalLayout_name_matrixpara->addWidget(label_name_Bmin);

        label_name_Bmax = new QLabel(gridLayoutWidget_3);
        label_name_Bmax->setObjectName(QString::fromUtf8("label_name_Bmax"));

        horizontalLayout_name_matrixpara->addWidget(label_name_Bmax);

        label_name_makematrix = new QLabel(gridLayoutWidget_3);
        label_name_makematrix->setObjectName(QString::fromUtf8("label_name_makematrix"));

        horizontalLayout_name_matrixpara->addWidget(label_name_makematrix);


        gridLayout_randommatrix->addLayout(horizontalLayout_name_matrixpara, 1, 0, 1, 1);

        horizontalLayout_name_randommatrix = new QHBoxLayout();
        horizontalLayout_name_randommatrix->setSpacing(6);
        horizontalLayout_name_randommatrix->setObjectName(QString::fromUtf8("horizontalLayout_name_randommatrix"));
        label_name_matrixmaker = new QLabel(gridLayoutWidget_3);
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

        horizontalLayout_name_randommatrix->addWidget(label_name_matrixmaker);


        gridLayout_randommatrix->addLayout(horizontalLayout_name_randommatrix, 0, 0, 1, 1);

        horizontalLayout_matrixpara = new QHBoxLayout();
        horizontalLayout_matrixpara->setSpacing(6);
        horizontalLayout_matrixpara->setObjectName(QString::fromUtf8("horizontalLayout_matrixpara"));
        spinBox_Arow = new QSpinBox(gridLayoutWidget_3);
        spinBox_Arow->setObjectName(QString::fromUtf8("spinBox_Arow"));
        spinBox_Arow->setMinimum(1);
        spinBox_Arow->setMaximum(10000);

        horizontalLayout_matrixpara->addWidget(spinBox_Arow);

        spinBox_sameside = new QSpinBox(gridLayoutWidget_3);
        spinBox_sameside->setObjectName(QString::fromUtf8("spinBox_sameside"));
        spinBox_sameside->setMinimum(1);
        spinBox_sameside->setMaximum(10000);

        horizontalLayout_matrixpara->addWidget(spinBox_sameside);

        spinBox_Bcol = new QSpinBox(gridLayoutWidget_3);
        spinBox_Bcol->setObjectName(QString::fromUtf8("spinBox_Bcol"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(spinBox_Bcol->sizePolicy().hasHeightForWidth());
        spinBox_Bcol->setSizePolicy(sizePolicy1);
        spinBox_Bcol->setMinimum(1);
        spinBox_Bcol->setMaximum(10000);

        horizontalLayout_matrixpara->addWidget(spinBox_Bcol);

        comboBox_Atype = new QComboBox(gridLayoutWidget_3);
        comboBox_Atype->addItem(QString());
        comboBox_Atype->addItem(QString());
        comboBox_Atype->addItem(QString());
        comboBox_Atype->addItem(QString());
        comboBox_Atype->setObjectName(QString::fromUtf8("comboBox_Atype"));
        comboBox_Atype->setEnabled(true);
        comboBox_Atype->setEditable(true);

        horizontalLayout_matrixpara->addWidget(comboBox_Atype);

        comboBox_Btype = new QComboBox(gridLayoutWidget_3);
        comboBox_Btype->addItem(QString());
        comboBox_Btype->addItem(QString());
        comboBox_Btype->addItem(QString());
        comboBox_Btype->addItem(QString());
        comboBox_Btype->setObjectName(QString::fromUtf8("comboBox_Btype"));
        comboBox_Btype->setEditable(true);

        horizontalLayout_matrixpara->addWidget(comboBox_Btype);

        doubleSpinBox_Amin = new QDoubleSpinBox(gridLayoutWidget_3);
        doubleSpinBox_Amin->setObjectName(QString::fromUtf8("doubleSpinBox_Amin"));

        horizontalLayout_matrixpara->addWidget(doubleSpinBox_Amin);

        doubleSpinBox_Amax = new QDoubleSpinBox(gridLayoutWidget_3);
        doubleSpinBox_Amax->setObjectName(QString::fromUtf8("doubleSpinBox_Amax"));

        horizontalLayout_matrixpara->addWidget(doubleSpinBox_Amax);

        doubleSpinBox_Bmin = new QDoubleSpinBox(gridLayoutWidget_3);
        doubleSpinBox_Bmin->setObjectName(QString::fromUtf8("doubleSpinBox_Bmin"));

        horizontalLayout_matrixpara->addWidget(doubleSpinBox_Bmin);

        doubleSpinBox_Bmax = new QDoubleSpinBox(gridLayoutWidget_3);
        doubleSpinBox_Bmax->setObjectName(QString::fromUtf8("doubleSpinBox_Bmax"));

        horizontalLayout_matrixpara->addWidget(doubleSpinBox_Bmax);

        pushButton_confirmmake = new QPushButton(gridLayoutWidget_3);
        pushButton_confirmmake->setObjectName(QString::fromUtf8("pushButton_confirmmake"));

        horizontalLayout_matrixpara->addWidget(pushButton_confirmmake);


        gridLayout_randommatrix->addLayout(horizontalLayout_matrixpara, 2, 0, 1, 1);

        gridLayoutWidget = new QWidget(centralWidget);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(610, 110, 591, 341));
        gridLayout_rescal = new QGridLayout(gridLayoutWidget);
        gridLayout_rescal->setSpacing(6);
        gridLayout_rescal->setContentsMargins(11, 11, 11, 11);
        gridLayout_rescal->setObjectName(QString::fromUtf8("gridLayout_rescal"));
        gridLayout_rescal->setContentsMargins(0, 0, 0, 0);
        tableView_showres = new QTableView(gridLayoutWidget);
        tableView_showres->setObjectName(QString::fromUtf8("tableView_showres"));

        gridLayout_rescal->addWidget(tableView_showres, 0, 0, 1, 1);

        QtMatrixOpenMPClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QtMatrixOpenMPClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1221, 26));
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

        comboBox_algoNormal->setItemText(0, QApplication::translate("QtMatrixOpenMPClass", "\346\234\264\347\264\240\347\237\251\351\230\265\344\271\230\346\263\225", nullptr));
        comboBox_algoNormal->setItemText(1, QApplication::translate("QtMatrixOpenMPClass", "\346\234\264\347\264\240\345\271\266\350\241\214\344\271\230\346\263\225", nullptr));
        comboBox_algoNormal->setItemText(2, QApplication::translate("QtMatrixOpenMPClass", "Strassen\347\256\227\346\263\225", nullptr));

        pushButton_insertqueque->setText(QApplication::translate("QtMatrixOpenMPClass", "\346\217\222\345\205\245\350\256\241\347\256\227\351\230\237\345\210\227", nullptr));
        pushButton_deleteque->setText(QApplication::translate("QtMatrixOpenMPClass", "\344\273\216\350\256\241\347\256\227\351\230\237\345\210\227\345\210\240\351\231\244", nullptr));
        pushButton_startcal->setText(QApplication::translate("QtMatrixOpenMPClass", "\345\274\200\345\247\213\350\256\241\347\256\227", nullptr));
        pushButton_clear->setText(QApplication::translate("QtMatrixOpenMPClass", "\345\205\250\351\203\250\346\270\205\347\251\272", nullptr));
        label_name_algoformer->setText(QApplication::translate("QtMatrixOpenMPClass", "\345\205\210\350\241\214\347\256\227\346\263\225\351\200\211\346\213\251", nullptr));
        label_name_algolatter->setText(QApplication::translate("QtMatrixOpenMPClass", "\345\220\216\347\275\256\347\256\227\346\263\225\351\200\211\346\213\251", nullptr));
        label_name_corenum->setText(QApplication::translate("QtMatrixOpenMPClass", "cpu\346\240\270\345\277\203\346\225\260", nullptr));
        label_name_ARow->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\350\241\214\346\225\260", nullptr));
        label_name_sameside->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\233\270\345\220\214\350\276\271\351\225\277", nullptr));
        label_name_Bcol->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\345\210\227\346\225\260", nullptr));
        label_name_Atype->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\347\261\273\345\236\213", nullptr));
        label_name_Btype->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\347\261\273\345\236\213", nullptr));
        label_name_Amin->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\346\234\200\345\260\217\345\200\274", nullptr));
        label_name_Amax->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265A\346\234\200\345\244\247\345\200\274", nullptr));
        label_name_Bmin->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\346\234\200\345\260\217\345\200\274", nullptr));
        label_name_Bmax->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\237\251\351\230\265B\346\234\200\345\244\247\345\200\274", nullptr));
        label_name_makematrix->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\224\237\346\210\220\347\237\251\351\230\265", nullptr));
        label_name_matrixmaker->setText(QApplication::translate("QtMatrixOpenMPClass", "\351\232\217\346\234\272\347\237\251\351\230\265\347\224\237\346\210\220\345\231\250", nullptr));
        comboBox_Atype->setItemText(0, QApplication::translate("QtMatrixOpenMPClass", "Integer", nullptr));
        comboBox_Atype->setItemText(1, QApplication::translate("QtMatrixOpenMPClass", "Float", nullptr));
        comboBox_Atype->setItemText(2, QApplication::translate("QtMatrixOpenMPClass", "Double", nullptr));
        comboBox_Atype->setItemText(3, QApplication::translate("QtMatrixOpenMPClass", "Long Long", nullptr));

        comboBox_Btype->setItemText(0, QApplication::translate("QtMatrixOpenMPClass", "Integer", nullptr));
        comboBox_Btype->setItemText(1, QApplication::translate("QtMatrixOpenMPClass", "Float", nullptr));
        comboBox_Btype->setItemText(2, QApplication::translate("QtMatrixOpenMPClass", "Double", nullptr));
        comboBox_Btype->setItemText(3, QApplication::translate("QtMatrixOpenMPClass", "Long Long", nullptr));

        pushButton_confirmmake->setText(QApplication::translate("QtMatrixOpenMPClass", "\347\241\256\350\256\244\347\224\237\346\210\220", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QtMatrixOpenMPClass: public Ui_QtMatrixOpenMPClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTMATRIXOPENMP_H
