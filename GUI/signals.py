from PyQt5.QtCore import QObject, pyqtSignal

class GlobalSignals(QObject):
    get_neuron_filters = pyqtSignal()
    send_neuron_filters = pyqtSignal(str, bool)
    send_recordings = pyqtSignal(list)

signals = GlobalSignals()