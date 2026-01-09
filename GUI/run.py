import os, re, pprint

print(f'current working directory: {os.getcwd()}')

from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QFileSystemModel, QTreeWidgetItem
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QColor, QBrush

import sys
import traceback
import subprocess, json


def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb)
    sys.__excepthook__(type, value, tb)

sys.excepthook = excepthook

def ui_path():
    here = os.path.dirname(os.path.abspath(__file__))  # .../GUI
    # When frozen, PyInstaller unpacks to sys._MEIPASS
    if getattr(sys, 'frozen', False):
        # support either destination you choose in --add-data (root or GUI/)
        candidates = [
            os.path.join(sys._MEIPASS, "GUI.ui"),
            os.path.join(sys._MEIPASS, "GUI", "GUI.ui"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
    return os.path.join(here, "GUI.ui")

Ui_MainWindow, QtBaseClass = uic.loadUiType(ui_path())
# qtCreatorFile = resource_path("GUI/GUI.ui")
# Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

from functions import MyFunctions
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.neural_data_folder = r'R:\Data\RhythmPerception\Neural Recordings\Recordings'

        # self.populate_files()

        self.myfuncs =  MyFunctions()

        # self.sortButton.clicked.connect(self.sort_button_clicked)

        # ===== DEFAULT CONDITIONS =====
        files = self.sort_files(self.neural_data_folder)
        for path, loc in files:
            name = path.split("\\")[-1]
            item = QTreeWidgetItem([name, loc])
            item.setData(0,Qt.UserRole,path)

            color = self.get_item_color(path)

            for col in range(item.columnCount()):
                item.setBackground(col, QBrush(color))

            if loc == 'NCM':
                self.treeViewNCM.addTopLevelItem(item)
            elif loc == 'Field L':
                self.treeViewFieldL.addTopLevelItem(item)
            elif loc == 'HVC':
                self.treeViewHVC.addTopLevelItem(item)
            elif loc == 'Area X':
                self.treeViewAreaX.addTopLevelItem(item)

        self.treeViewNCM.itemClicked.connect(self.clicked)
        self.treeViewFieldL.itemClicked.connect(self.clicked)
        self.treeViewHVC.itemClicked.connect(self.clicked)
        self.treeViewAreaX.itemClicked.connect(self.clicked)

        # ===== SIGNALS =====

    def get_item_color(self, path):
        color = 0

        sorting_folder_path = os.path.join(path, 'sorting', 'sorting_TDC')
        gui_output_path = os.path.join(path, "sorting", "analyzer_TDC_binary", "spikeinterface_gui")

        if os.path.exists(sorting_folder_path):
            color += 1
        if os.path.exists(gui_output_path):
            color += 1

        if color == 2:
            return QColor(210, 245, 210)
        elif color == 1:
            return QColor(255, 235, 200)
        elif color == 0:
            return QColor(238, 238, 238)
        else:
            return QColor(255, 215, 215)


    def sort_files(self, base_dir):
        pattern = re.compile(r'^([A-Za-z]\d{2}[A-Za-z]\d{2}|[A-Za-z]{2}\d{2}[A-Za-z]{2}\d{2})$')
        children = [
            name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and pattern.match(name)
        ]

        recordings = []
        for name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, name)) and pattern.match(name):
                recs = os.listdir(os.path.join(base_dir, name))
                for r in recs:
                    if 'ncm' in r.lower():
                        loc = ('NCM')
                    elif 'field l' in r.lower() or 'fieldl' in r.lower():
                        loc = 'Field L'
                    elif 'hvc' in r.lower():
                        loc = 'HVC'
                    elif 'areax' in r.lower() or 'area x' in r.lower():
                        loc = 'Area X'
                    elif 'cm' in r.lower() or 'cmm' in r.lower():
                        loc = 'CM'
                    else:
                        loc = None

                    if loc is not None:
                        recordings.append((os.path.join(base_dir, name, r), loc))
        return recordings

    def clicked(self, item, column):
        path = item.data(0, Qt.UserRole)
        if path:
            self.myfuncs.file_clicked(path)
if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication(sys.argv)
        ui = MainWindow()
        ui.show()
        sys.exit(app.exec_())
    finally:
        print("Shutting down.")