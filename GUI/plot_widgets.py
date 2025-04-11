# PySide6 (GUI)
from PySide6.QtWidgets import QWidget, QVBoxLayout

# Matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Criar classe para o gráfico Matplotlib
class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Cria uma figura do Matplotlib
        self.figure = Figure(figsize=(12, 9), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Cria os eixos 3D e armazena uma referência
        self.ax = self.figure.add_subplot(111, projection='3d')

        # Adiciona o canvas ao layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)