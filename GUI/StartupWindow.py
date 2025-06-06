from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QRadioButton, QLineEdit,
    QPushButton, QGridLayout, QVBoxLayout, QGroupBox, QButtonGroup, QMessageBox
)
from PySide6.QtGui import QFont, QPixmap, QIcon
from PySide6.QtCore import Qt, Signal
import sys
import ctypes

class SegundaTela(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configurações da Estrutura")
        self.setGeometry(150, 150, 600, 400)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        label = QLabel("🔧 Configurações da Estrutura (futura etapa)")
        label.setFont(QFont("Arial", 14))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        self.setLayout(layout)

class SimuFrameWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SimuFrame - Simulation Frame")
        self.setWindowIcon(QIcon("GUI/icone.ico"))
        self.setGeometry(100, 100, 500, 300)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Background image
        logo = QLabel()
        logo.setPixmap(QPixmap("GUI/icone.ico").scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(logo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Structure configuration 
        estrutura_box = QGroupBox("Seleção da estrutura")
        estrutura_box.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
            }
        """)
        estrutura_layout = QVBoxLayout()

        # Structure selection (beam, column, etc.)
        self.combo_estrutura = QComboBox()
        self.combo_estrutura.setMinimumWidth(200)
        self.combo_estrutura.setStyleSheet("""
            QComboBox {
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        self.combo_estrutura.addItem("Selecione o tipo de estrutura...")
        self.combo_estrutura.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)
        self.combo_estrutura.addItems(["Viga", "Pilar", "Pórtico", "Treliça"])
        self.combo_estrutura.currentTextChanged.connect(self.atualizar_subtipos)
        estrutura_layout.addWidget(self.combo_estrutura)

        # Structure subtype layout
        self.subtipo_layout = QGridLayout()

        # Structure subtype selection
        self.combo_subtipo = QComboBox()
        self.combo_subtipo.setMinimumWidth(150)
        self.combo_subtipo.setStyleSheet("""
            QComboBox {
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 4px;
                font-size: 14px;
            }
        """)
        self.combo_subtipo.setVisible(False)

        # Number of subdivisions
        self.subdivisoes = QLineEdit()
        self.subdivisoes.setPlaceholderText("N. de Subdivisões")
        self.subdivisoes.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 4px;
                font-size: 14px;
            }
        """)
        self.subdivisoes.setVisible(False)

        # Preview button
        self.btn_preview = QPushButton("🔎 Pré-Visualizar")
        self.btn_preview.setMinimumWidth(150)
        self.btn_preview.setStyleSheet("""
            QPushButton {
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
                alignment: center;
            }
            QPushButton:hover {
                background-color: #bbb;
            }
        """)
        self.btn_preview.setVisible(False)
        self.btn_preview.clicked.connect(self.previsualizar_estrutura)

        # Add to layout
        self.subtipo_layout.addWidget(self.combo_subtipo, 0, 0)
        self.subtipo_layout.addWidget(self.subdivisoes, 0, 1)
        self.subtipo_layout.addWidget(self.btn_preview, 0, 2)
        estrutura_layout.addLayout(self.subtipo_layout)
        estrutura_box.setLayout(estrutura_layout)
        main_layout.addWidget(estrutura_box)

        # Add to main layout
        self.setLayout(main_layout)

        # Analysis selection (linear, non-linear or buckling)
        analise_box = QGroupBox("Tipo de Análise")
        analise_box.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
            }
        """)
        analise_layout = QVBoxLayout()
        self.radio_linear = QRadioButton("Análise Linear")
        self.radio_linear.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
            }
        """)
        self.radio_nao_linear = QRadioButton("Análise Não Linear (Geométrica)")
        self.radio_nao_linear.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
            }
        """)
        self.radio_flambagem = QRadioButton("Flambagem Linear")
        self.radio_flambagem.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
            }
        """)
        self.radio_linear.setChecked(True)

        self.tipo_analise_group = QButtonGroup()
        self.tipo_analise_group.addButton(self.radio_linear, id=0)
        self.tipo_analise_group.addButton(self.radio_nao_linear, id=1)
        self.tipo_analise_group.addButton(self.radio_flambagem, id=2)

        analise_layout.addWidget(self.radio_linear)
        analise_layout.addWidget(self.radio_nao_linear)
        analise_layout.addWidget(self.radio_flambagem)
        analise_box.setLayout(analise_layout)
        main_layout.addWidget(analise_box)

        # Botão de iniciar
        self.btn_iniciar = QPushButton("📊 Iniciar Análise")
        self.btn_iniciar.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57;
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #276C4E;
            }
        """)
        self.btn_iniciar.clicked.connect(self.abrir_segunda_tela)
        main_layout.addWidget(self.btn_iniciar, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(main_layout)

    def atualizar_subtipos(self, tipo):
        subtipos = {
            "Viga": ["Exemplo 1", "Exemplo 2", "Exemplo 3", "Exemplo 4"],
            "Pilar": ["Exemplo 1"],
            "Pórtico": ["Exemplo 1", "Exemplo 2", "Exemplo 3", "Exemplo 4"],
            "Treliça": ["Exemplo 1", "Exemplo 2", "Exemplo 3", "Exemplo 4"]
        }

        if tipo in subtipos:
            self.combo_subtipo.clear()
            self.combo_subtipo.addItems(subtipos[tipo])
            self.combo_subtipo.setVisible(True)
            self.subdivisoes.setVisible(True)
            self.btn_preview.setVisible(True)
        else:
            self.combo_subtipo.setVisible(False)
            self.subdivisoes.setVisible(False)
            self.btn_preview.setVisible(False)
    
    # Signal to get the selected structure
    estruturaSelecionada = Signal(str, str, int, int, bool)

    def emitir_estrutura(self, plot):
        tipo = self.combo_estrutura.currentText()
        subtipo = self.combo_subtipo.currentIndex()

        # Get the number of subdivisions
        try:
            n = int(self.subdivisoes.text())
        except ValueError:
            QMessageBox.critical(self, "Erro", "Por favor, insira um número de subdivisões válido (ex.: 5).")
            return
        
        # Get the selected analysis
        selected_id = self.tipo_analise_group.checkedId()
        if selected_id == 0:
            analise = "linear"
        elif selected_id == 1:
            analise = "nao-linear"
        elif selected_id == 2:
            analise = "flambagem"

        self.estruturaSelecionada.emit(analise, tipo, subtipo, n, plot)
        return True
    

    def previsualizar_estrutura(self):
        # Emit the signal
        self.emitir_estrutura(plot=True)

    def abrir_segunda_tela(self):
        # Emit the signal
        success = self.emitir_estrutura(plot=False)

        # Verify if the signal was successful
        if not success:
            return

        # Start analysis
        self.segunda_tela = SegundaTela()
        self.segunda_tela.show()

# Definir o identificador do aplicativo
myappid = 'simuframe.app.0.1'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon = QIcon("GUI/icone.ico")
    app.setWindowIcon(icon)
    janela = SimuFrameWindow()
    janela.show()
    sys.exit(app.exec())
