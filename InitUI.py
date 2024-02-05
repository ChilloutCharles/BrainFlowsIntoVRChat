from multiprocessing import Value
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from enum import Enum
import subprocess
import main

class hardwareList(Enum):

    NO_BOARD = -100
    PLAYBACK_FILE = -3
    STREAMING = -2
    SYNTHETIC = -1
    CYTON = 0
    GANGLION = 1
    CYTON_DAISY = 2
    GALEA = 3
    GANGLION_WIFI = 4
    CYTON_WIFI = 5
    CYTON_DAISY_WIFI = 6
    BRAINBIT = 7
    UNICORN = 8
    CALLIBRI_EEG = 9
    CALLIBRI_EMG = 10  
    CALLIBRI_ECG = 11  
    NOTION_1 = 13  
    NOTION_2 = 14  
    GFORCE_PRO = 16  
    FREEEEG32 = 17  
    BRAINBIT_BLED = 18  
    GFORCE_DUAL = 19  
    GALEA_SERIAL = 20  
    MUSE_S_BLED = 21  
    MUSE_2_BLED = 22  
    CROWN = 23  
    ANT_NEURO_EE_410 = 24  
    ANT_NEURO_EE_411 = 25  
    ANT_NEURO_EE_430 = 26  
    ANT_NEURO_EE_211 = 27  
    ANT_NEURO_EE_212 = 28  
    ANT_NEURO_EE_213 = 29  
    ANT_NEURO_EE_214 = 30  
    ANT_NEURO_EE_215 = 31  
    ANT_NEURO_EE_221 = 32  
    ANT_NEURO_EE_222 = 33  
    ANT_NEURO_EE_223 = 34  
    ANT_NEURO_EE_224 = 35  
    ANT_NEURO_EE_225 = 36  
    ENOPHONE = 37  
    MUSE_2 = 38  
    MUSE_S = 39  
    BRAINALIVE = 40  
    MUSE_2016 = 41  
    MUSE_2016_BLED = 42  
    EXPLORE_4_CHAN = 44  
    EXPLORE_8_CHAN = 45  
    GANGLION_NATIVE = 46  
    EMOTIBIT = 47  
    GALEA_V4 = 48  
    GALEA_SERIAL_V4 = 49  
    NTL_WIFI = 50  
    ANT_NEURO_EE_511 = 51  
    FREEEEG128 = 52  
    AAVAA_V3 = 53  
    
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(700, 215)
        MainWindow.setMouseTracking(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.CB1_Hardware = QComboBox(self.centralwidget)
        self.CB1_Hardware.addItem(u"NO_BOARD")
        self.CB1_Hardware.addItem(u"PLAYBACK_FILE")
        self.CB1_Hardware.addItem(u"STREAMING")
        self.CB1_Hardware.addItem(u"SYNTHETIC")
        self.CB1_Hardware.addItem(u"CYTON")
        self.CB1_Hardware.addItem(u"GANGLION")
        self.CB1_Hardware.addItem(u"CYTON_DAISY")
        self.CB1_Hardware.addItem(u"GALEA")
        self.CB1_Hardware.addItem(u"GANGLION_WIFI")
        self.CB1_Hardware.addItem(u"CYTON_WIFI")
        self.CB1_Hardware.addItem(u"CYTON_DAISY_WIFI")
        self.CB1_Hardware.addItem(u"BRAINBIT")
        self.CB1_Hardware.addItem(u"UNICORN")
        self.CB1_Hardware.addItem(u"CALLIBRI_EEG")
        self.CB1_Hardware.addItem(u"CALLIBRI_EMG")
        self.CB1_Hardware.addItem(u"CALLIBRI_ECG")
        self.CB1_Hardware.addItem(u"NOTION_1")
        self.CB1_Hardware.addItem(u"NOTION_2")
        self.CB1_Hardware.addItem(u"GFORCE_PRO")
        self.CB1_Hardware.addItem(u"FREEEEG32")
        self.CB1_Hardware.addItem(u"BRAINBIT_BLED")
        self.CB1_Hardware.addItem(u"GFORCE_DUAL")
        self.CB1_Hardware.addItem(u"GALEA_SERIAL")
        self.CB1_Hardware.addItem(u"MUSE_S_BLED")
        self.CB1_Hardware.addItem(u"MUSE_2_BLED")
        self.CB1_Hardware.addItem(u"CROWN")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_410")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_411")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_430")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_211")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_212")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_213")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_214")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_215")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_221")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_222")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_223")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_224")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_225")
        self.CB1_Hardware.addItem(u"ENOPHONE")
        self.CB1_Hardware.addItem(u"MUSE_2")
        self.CB1_Hardware.addItem(u"MUSE_S")
        self.CB1_Hardware.addItem(u"BRAINALIVE")
        self.CB1_Hardware.addItem(u"MUSE_2016")
        self.CB1_Hardware.addItem(u"MUSE_2016_BLED")
        self.CB1_Hardware.addItem(u"EXPLORE_4_CHAN")
        self.CB1_Hardware.addItem(u"EXPLORE_8_CHAN")
        self.CB1_Hardware.addItem(u"GANGLION_NATIVE")
        self.CB1_Hardware.addItem(u"EMOTIBIT")
        self.CB1_Hardware.addItem(u"GALEA_V4")
        self.CB1_Hardware.addItem(u"GALEA_SERIAL_V4")
        self.CB1_Hardware.addItem(u"NTL_WIFI")
        self.CB1_Hardware.addItem(u"ANT_NEURO_EE_511")
        self.CB1_Hardware.addItem(u"FREEEEG128")
        self.CB1_Hardware.addItem(u"AAVAA_V3")
        self.CB1_Hardware.setObjectName(u"CB1_Hardware")
        self.CB1_Hardware.setGeometry(QRect(550, 90, 129, 20))
        self.CB1_Hardware.setMouseTracking(False)
        self.CB1_Hardware.setEditable(False)
        self.CB1_Hardware.setMaxVisibleItems(20)
        self.CB1_Hardware.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.B1_StartTracking = QPushButton(self.centralwidget)
        self.B1_StartTracking.setObjectName(u"B1_StartTracking")
        self.B1_StartTracking.setGeometry(QRect(550, 20, 131, 61))
        self.TF9_Timeout = QTextEdit(self.centralwidget)
        self.TF9_Timeout.setObjectName(u"TF9_Timeout")
        self.TF9_Timeout.setGeometry(QRect(665, 150, 31, 23))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TF9_Timeout.sizePolicy().hasHeightForWidth())
        self.TF9_Timeout.setSizePolicy(sizePolicy)
        font = QFont()
        font.setKerning(False)
        self.TF9_Timeout.setFont(font)
        self.TF9_Timeout.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF9_Timeout.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF9_Timeout.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.TF9_Timeout.setLineWrapMode(QTextEdit.NoWrap)
        self.TF9_Timeout.setLineWrapColumnOrWidth(0)
        self.TF9_Timeout.setMarkdown(u"")
        self.TF9_Timeout.setPlaceholderText(u"")
        self.L1_Timeout = QLabel(self.centralwidget)
        self.L1_Timeout.setObjectName(u"L1_Timeout")
        self.L1_Timeout.setGeometry(QRect(560, 150, 101, 23))
        self.TF1_IP = QTextEdit(self.centralwidget)
        self.TF1_IP.setObjectName(u"TF1_IP")
        self.TF1_IP.setGeometry(QRect(180, 15, 92, 23))
        sizePolicy.setHeightForWidth(self.TF1_IP.sizePolicy().hasHeightForWidth())
        self.TF1_IP.setSizePolicy(sizePolicy)
        self.TF1_IP.setFont(font)
        self.TF1_IP.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF1_IP.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF1_IP.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.TF1_IP.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF1_IP.setLineWrapColumnOrWidth(0)
        self.TF1_IP.setMarkdown(u"")
        self.TF1_IP.setPlaceholderText(u"")
        self.TF2_Port = QTextEdit(self.centralwidget)
        self.TF2_Port.setObjectName(u"TF2_Port")
        self.TF2_Port.setGeometry(QRect(270, 15, 41, 23))
        sizePolicy.setHeightForWidth(self.TF2_Port.sizePolicy().hasHeightForWidth())
        self.TF2_Port.setSizePolicy(sizePolicy)
        self.TF2_Port.setFont(font)
        self.TF2_Port.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF2_Port.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF2_Port.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.TF2_Port.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF2_Port.setLineWrapColumnOrWidth(0)
        self.TF2_Port.setMarkdown(u"")
        self.TF2_Port.setPlaceholderText(u"")
        self.TF3_MAC = QTextEdit(self.centralwidget)
        self.TF3_MAC.setObjectName(u"TF3_MAC")
        self.TF3_MAC.setGeometry(QRect(120, 45, 121, 23))
        sizePolicy.setHeightForWidth(self.TF3_MAC.sizePolicy().hasHeightForWidth())
        self.TF3_MAC.setSizePolicy(sizePolicy)
        self.TF3_MAC.setFont(font)
        self.TF3_MAC.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF3_MAC.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF3_MAC.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.TF3_MAC.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF3_MAC.setLineWrapColumnOrWidth(0)
        self.TF3_MAC.setMarkdown(u"")
        self.TF3_MAC.setPlaceholderText(u"")
        self.CB2_IPConfig = QComboBox(self.centralwidget)
        self.CB2_IPConfig.addItem("")
        self.CB2_IPConfig.addItem(u"UDP")
        self.CB2_IPConfig.addItem(u"TCP")
        self.CB2_IPConfig.setObjectName(u"CB2_IPConfig")
        self.CB2_IPConfig.setGeometry(QRect(120, 15, 61, 23))
        self.TF4_COM = QTextEdit(self.centralwidget)
        self.TF4_COM.setObjectName(u"TF4_COM")
        self.TF4_COM.setGeometry(QRect(120, 75, 61, 23))
        sizePolicy.setHeightForWidth(self.TF4_COM.sizePolicy().hasHeightForWidth())
        self.TF4_COM.setSizePolicy(sizePolicy)
        self.TF4_COM.setFont(font)
        self.TF4_COM.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF4_COM.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF4_COM.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.TF4_COM.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF4_COM.setLineWrapColumnOrWidth(0)
        self.TF4_COM.setMarkdown(u"")
        self.TF4_COM.setPlaceholderText(u"")
        self.TF7_Misc = QTextEdit(self.centralwidget)
        self.TF7_Misc.setObjectName(u"TF7_Misc")
        self.TF7_Misc.setGeometry(QRect(120, 180, 286, 23))
        sizePolicy.setHeightForWidth(self.TF7_Misc.sizePolicy().hasHeightForWidth())
        self.TF7_Misc.setSizePolicy(sizePolicy)
        self.TF7_Misc.setFont(font)
        self.TF7_Misc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF7_Misc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF7_Misc.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.TF7_Misc.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF7_Misc.setLineWrapColumnOrWidth(0)
        self.TF7_Misc.setMarkdown(u"")
        self.TF7_Misc.setPlaceholderText(u"")
        self.TF6_LaunchParm = QTextEdit(self.centralwidget)
        self.TF6_LaunchParm.setObjectName(u"TF6_LaunchParm")
        self.TF6_LaunchParm.setGeometry(QRect(120, 150, 286, 23))
        sizePolicy.setHeightForWidth(self.TF6_LaunchParm.sizePolicy().hasHeightForWidth())
        self.TF6_LaunchParm.setSizePolicy(sizePolicy)
        self.TF6_LaunchParm.setFont(font)
        self.TF6_LaunchParm.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF6_LaunchParm.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF6_LaunchParm.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.TF6_LaunchParm.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF6_LaunchParm.setLineWrapColumnOrWidth(0)
        self.TF6_LaunchParm.setMarkdown(u"")
        self.TF6_LaunchParm.setPlaceholderText(u"")
        self.TF5_SN = QTextEdit(self.centralwidget)
        self.TF5_SN.setObjectName(u"TF5_SN")
        self.TF5_SN.setGeometry(QRect(120, 120, 151, 23))
        sizePolicy.setHeightForWidth(self.TF5_SN.sizePolicy().hasHeightForWidth())
        self.TF5_SN.setSizePolicy(sizePolicy)
        self.TF5_SN.setFont(font)
        self.TF5_SN.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF5_SN.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF5_SN.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.TF5_SN.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF5_SN.setLineWrapColumnOrWidth(0)
        self.TF5_SN.setMarkdown(u"")
        self.TF5_SN.setPlaceholderText(u"")
        self.TF8_LoadCSV = QTextEdit(self.centralwidget)
        self.TF8_LoadCSV.setObjectName(u"TF8_LoadCSV")
        self.TF8_LoadCSV.setGeometry(QRect(420, 180, 241, 23))
        sizePolicy.setHeightForWidth(self.TF8_LoadCSV.sizePolicy().hasHeightForWidth())
        self.TF8_LoadCSV.setSizePolicy(sizePolicy)
        self.TF8_LoadCSV.setFont(font)
        self.TF8_LoadCSV.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF8_LoadCSV.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.TF8_LoadCSV.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.TF8_LoadCSV.setLineWrapMode(QTextEdit.WidgetWidth)
        self.TF8_LoadCSV.setLineWrapColumnOrWidth(0)
        self.TF8_LoadCSV.setMarkdown(u"")
        self.TF8_LoadCSV.setPlaceholderText(u"")
        self.TB3_Serial = QCheckBox(self.centralwidget)
        self.TB3_Serial.setObjectName(u"TB3_Serial")
        self.TB3_Serial.setGeometry(QRect(15, 75, 70, 23))
        self.TB1_IP = QCheckBox(self.centralwidget)
        self.TB1_IP.setObjectName(u"TB1_IP")
        self.TB1_IP.setGeometry(QRect(15, 15, 70, 23))
        self.TB2_MAC = QCheckBox(self.centralwidget)
        self.TB2_MAC.setObjectName(u"TB2_MAC")
        self.TB2_MAC.setGeometry(QRect(15, 45, 91, 23))
        self.TB5_LaunchParams = QCheckBox(self.centralwidget)
        self.TB5_LaunchParams.setObjectName(u"TB5_LaunchParams")
        self.TB5_LaunchParams.setGeometry(QRect(15, 150, 106, 23))
        self.TB7_LoadCSV = QCheckBox(self.centralwidget)
        self.TB7_LoadCSV.setObjectName(u"TB7_LoadCSV")
        self.TB7_LoadCSV.setGeometry(QRect(420, 150, 121, 23))
        self.TB4_ProdNum = QCheckBox(self.centralwidget)
        self.TB4_ProdNum.setObjectName(u"TB4_ProdNum")
        self.TB4_ProdNum.setGeometry(QRect(15, 120, 101, 23))
        self.TB6_Other = QCheckBox(self.centralwidget)
        self.TB6_Other.setObjectName(u"TB6_Other")
        self.TB6_Other.setGeometry(QRect(15, 180, 76, 23))
        self.EX1_fileMng = QToolButton(self.centralwidget)
        self.EX1_fileMng.setObjectName(u"EX1_fileMng")
        self.EX1_fileMng.setGeometry(QRect(670, 180, 25, 23))
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)

        self.CB1_Hardware.setCurrentIndex(40)
        def hardwareID():
            FirstSwitch = self.CB1_Hardware.currentText()
            SecondSwitch = hardwareList[FirstSwitch].value
            global updateHWID
            updateHWID = str(SecondSwitch) 
            print (FirstSwitch, 'has been selected as your EEG hardware.')
        hardwareID ()
        
        self.CB1_Hardware.currentTextChanged.connect(hardwareID)             
        helpCMD = "--help"
        def brainExec():
            subprocess.run(['main.py', '--board-id', updateHWID], shell=True)
        self.B1_StartTracking.clicked.connect(brainExec)



        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))

        self.CB1_Hardware.setCurrentText(QCoreApplication.translate("MainWindow", u"CYTON", None))
        self.B1_StartTracking.setText(QCoreApplication.translate("MainWindow", u"Start Tracking", None))
        self.L1_Timeout.setText(QCoreApplication.translate("MainWindow", u"Connection Timeout", None))
        self.CB2_IPConfig.setItemText(0, QCoreApplication.translate("MainWindow", u"NONE", None))

        self.TB3_Serial.setText(QCoreApplication.translate("MainWindow", u"Serial Port", None))
        self.TB1_IP.setText(QCoreApplication.translate("MainWindow", u"IP Config", None))
        self.TB2_MAC.setText(QCoreApplication.translate("MainWindow", u"MAC Address", None))
        self.TB5_LaunchParams.setText(QCoreApplication.translate("MainWindow", u"Launch Params", None))
        self.TB7_LoadCSV.setText(QCoreApplication.translate("MainWindow", u"Load Tracking Config", None))
        self.TB4_ProdNum.setText(QCoreApplication.translate("MainWindow", u"Serial Number", None))
        self.TB6_Other.setText(QCoreApplication.translate("MainWindow", u"Other Info", None))
        self.EX1_fileMng.setText(QCoreApplication.translate("MainWindow", u"...", None))
    # retranslateUi



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi (MainWindow)
    MainWindow.show()
    sys.exit (app.exec_())