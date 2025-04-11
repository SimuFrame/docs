# Importações iniciais
import sys
import numpy as np

# Bibliotecas autorais
from GUI.plot_widgets import MatplotlibWidget
from Visualização import plotar_deslocamentos, plotar_esforcos

# PySide6 (GUI)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QCheckBox, QLineEdit, QPushButton, QLabel,
    QGroupBox, QSizePolicy, QComboBox, QStackedWidget, QMessageBox, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from pyvistaqt import QtInteractor

def ValidacaoEstrutura(nome_estrutura):
    window = QMainWindow()

    # Verificar se já existe uma instância de QApplication
    app = QApplication.instance()
    if not app:  # Criar uma instância apenas se ela não existir
        app = QApplication(sys.argv)

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Validação da Estrutura")
    msg_box.setText(f"<p style='text-align: center; font-size: 16px;'>"
                    f"A estrutura <i>{nome_estrutura}</i> está correta?</p>")
    msg_box.setWindowIcon(QIcon(r"C:\Users\volpi\OneDrive\Documentos\Faculdade\MEF\Ícone.jpg"))
    msg_box.setIcon(QMessageBox.Icon.Question)

    # Adicionar botões "Sim" e "Não"
    botao_sim = msg_box.addButton("Sim", QMessageBox.ButtonRole.AcceptRole)
    botao_nao = msg_box.addButton("Não", QMessageBox.ButtonRole.RejectRole)

    # Ajustar estilo e centralizar a mensagem
    msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    # Exibir a caixa de diálogo
    msg_box.exec()

    # Verificar a resposta do usuário
    if msg_box.clickedButton() == botao_sim:
        msg_box.close()
    elif msg_box.clickedButton() == botao_nao:
        QMessageBox.warning(window, "Atenção", "<p style='font-size: 16px;'>Refaça a estrutura.</p")
        sys.exit()  # Encerra o programa

    # Finalizar o loop da aplicação apenas se ela foi criada neste escopo
    if not QApplication.instance():
        sys.exit(app.exec())


# Criar interface gráfica
class MainWindow(QMainWindow):
    def __init__(self, nome_estrutura, elementos, estrutura, tubos_ind, malha_ind, tubos_def, malha_def, num_modos, 
                 esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear,
                 desl_flambagem, xp, coords, MT, autovalores, pontos_int, coords_deformadas):
        super().__init__()
        self.setWindowTitle(f"SimuFrame: {nome_estrutura}")
        self.setWindowIcon(QIcon(r"C:\Users\volpi\Documents\Faculdade\MEF\Imagens TCC\TrussDeslLinear.png"))
        self.setGeometry(100, 100, 1200, 720)

        # Layout principal (horizontal)
        main_layout = QHBoxLayout()

        # Layout à esquerda (menu e controles)
        left_layout = QVBoxLayout()

        # Cria as abas e adiciona ao layout à esquerda
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_esforcos_tab(), "Esforços")
        self.tabs.addTab(self.create_displacement_tab(num_modos), "Deformação")
        self.tabs.addTab(self.create_comparison_tab(esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear), "Comparação de resultados")
        left_layout.addWidget(self.tabs)

        # Adiciona os controles de escala e botões ao layout à esquerda
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Escala do gráfico:")
        scale_label.setStyleSheet("font-size: 16px;")
        scale_layout.addWidget(scale_label)
        self.scale_input_esforcos = QLineEdit()
        self.scale_input_esforcos.setFixedHeight(30)
        self.scale_input_esforcos.setStyleSheet("font-size: 16px;")
        scale_layout.addWidget(self.scale_input_esforcos)
        left_layout.addLayout(scale_layout)

        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Aplicar")
        self.apply_button.clicked.connect(
            lambda: self.on_apply_button_clicked(elementos, estrutura, tubos_ind, malha_ind, tubos_def, malha_def, xp, coords,
                                                 esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear, desl_flambagem,
                                                 autovalores, MT, pontos_int, coords_deformadas))
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.close)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        left_layout.addLayout(button_layout)

        # Cria um widget para o layout esquerdo e define largura fixa
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400)  # Define a largura fixa do widget

        # Adiciona o layout à esquerda ao layout principal
        main_layout.addWidget(left_widget)

        # Cria o QStackedWidget para a área de plotagem à direita
        self.stacked_widget = QStackedWidget()

        # Adiciona widgets vazios para Matplotlib e PyVista
        self.matplotlib_widget = MatplotlibWidget()
        self.pyvista_widget = QtInteractor()

        self.stacked_widget.addWidget(self.matplotlib_widget)
        self.stacked_widget.addWidget(self.pyvista_widget)

        # Adiciona o QStackedWidget ao layout principal
        main_layout.addWidget(self.stacked_widget)

        # Define o layout principal
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_checkbox_with_label(self, checkbox, label_text):
        # Função auxiliar para criar um layout horizontal com um QCheckBox e QLabel
        h_layout = QHBoxLayout()
        h_layout.addWidget(checkbox)
        h_layout.addWidget(QLabel(label_text))
        h_layout.itemAt(1).widget().setStyleSheet("font-size: 16px;")
        h_layout.addStretch(1)

        return h_layout

    def on_checkbox_toggled(self, selected_checkbox, checkboxes):
        # Função genérica para garantir seleção exclusiva em um grupo de checkboxes
        if selected_checkbox.isChecked():
            for checkbox in checkboxes:
                if checkbox != selected_checkbox:
                    checkbox.setChecked(False)
    
    def create_esforcos_tab(self):
        # Cria a aba de Esforços com caixas de seleção e campo de escala
        widget = QWidget()
        layout = QVBoxLayout()

        # Menu de seleção para tipo de visualização
        visualizacao_label = QLabel("Tipo de Visualização:")
        visualizacao_label.setStyleSheet("font-size: 18px;")
        visualizacao_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.visualizacao_combo = QComboBox()
        self.visualizacao_combo.addItems(["Diagrama (Matplotlib)", "Colormap (PyVista)"])
        self.visualizacao_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.visualizacao_combo.setStyleSheet("font-size: 16px;")
        self.visualizacao_combo.setFixedHeight(30)
        layout.addWidget(visualizacao_label)
        layout.addWidget(self.visualizacao_combo)

        # Menu de seleção para tipo de esforços
        esforcos_label = QLabel("Tipo de Esforços:")
        esforcos_label.setStyleSheet("font-size: 18px;")
        esforcos_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.esforcos_combo = QComboBox()
        self.esforcos_combo.addItems(["Esforços lineares", "Esforços não-lineares"])
        self.esforcos_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.esforcos_combo.setStyleSheet("font-size: 16px;")
        self.esforcos_combo.setFixedHeight(30)
        layout.addWidget(esforcos_label)
        layout.addWidget(self.esforcos_combo)

        # Grupo de Esforços
        ntm_group = QGroupBox("Esforços")
        ntm_group.setStyleSheet("QGroupBox { font-size: 18px; }")
        ntm_layout = QVBoxLayout()

        # Criação das checkboxes
        self.fx_checkbox = QCheckBox()
        self.fy_checkbox = QCheckBox()
        self.fz_checkbox = QCheckBox()
        self.mx_checkbox = QCheckBox()
        self.my_checkbox = QCheckBox()
        self.mz_checkbox = QCheckBox()

        # Lista de checkboxes para esforços
        esforcos_checkboxes = [
            self.fx_checkbox, self.fy_checkbox, self.fz_checkbox,
            self.mx_checkbox, self.my_checkbox, self.mz_checkbox
        ]

        # Conecta cada checkbox à função on_checkbox_toggled
        for checkbox in esforcos_checkboxes:
            checkbox.toggled.connect(lambda checked, cb=checkbox: self.on_checkbox_toggled(cb, esforcos_checkboxes))

        # Adiciona cada par de checkbox e label ao layout de tensões
        labels = [
            ("F<sub>X</sub> - Translação (x)", self.fx_checkbox),
            ("F<sub>Y</sub> - Translação (y)", self.fy_checkbox),
            ("F<sub>Z</sub> - Translação (z)", self.fz_checkbox),
            ("M<sub>X</sub> - Momento torsor", self.mx_checkbox),
            ("M<sub>Y</sub> - Momento fletor (y)", self.my_checkbox),
            ("M<sub>Z</sub> - Momento fletor (z)", self.mz_checkbox)
        ]

        for label_text, checkbox in labels:
            ntm_layout.addLayout(self.create_checkbox_with_label(checkbox, label_text))

        ntm_group.setLayout(ntm_layout)
        layout.addWidget(ntm_group)

        widget.setLayout(layout)
        return widget

    def esforcos(self, elementos, estrutura, tubos_ind, malha_ind, tubos_def, malha_def, xp, 
                 coords, esforcos_lineares, esforcos_nao_lineares, MT, pontos_int, escala):
        visualizacao = self.visualizacao_combo.currentText()
        tipo_analise = self.esforcos_combo.currentText()

        # Determinar o esforço
        esforco = None
        for cb, esf in zip([self.fx_checkbox, self.fy_checkbox, self.fz_checkbox, self.mx_checkbox, self.my_checkbox, self.mz_checkbox],
                          ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']):
            if cb.isChecked():
                esforco = esf
                break

        # Verificar se uma checkbox foi selecionada
        if esforco is None:
            print("Por favor, selecione uma checkbox.")
            return
        
        # Determinar o tipo de análise
        analise = "linear" if tipo_analise == "Esforços lineares" else "não-linear"

        # Chamar a função de plotagem com os tipos selecionados
        if visualizacao == "Diagrama (Matplotlib)":
            self.stacked_widget.setCurrentIndex(0)
            biblioteca, widget = 'matplotlib', self.matplotlib_widget.figure
        elif visualizacao == "Colormap (PyVista)":
            self.stacked_widget.setCurrentIndex(1)
            biblioteca, widget = 'pyvista', self.pyvista_widget
        else:
            print("Tipo de visualização inválido.")
            return

        plotar_esforcos(tubos_ind, malha_ind, tubos_def, malha_def, elementos, estrutura, xp, coords,
                        esforcos_lineares, esforcos_nao_lineares, MT, pontos_int, biblioteca, analise, esforco, escala, widget)

    def create_displacement_tab(self, num_modos):
        # Cria a aba de Deformação com caixas de seleção e campo de escala
        widget = QWidget()
        layout = QVBoxLayout()

        # Grupo de Deslocamentos Nodais
        displacement_group = QGroupBox("Deslocamentos Nodais")
        displacement_group.setStyleSheet("QGroupBox { font-size: 18px; }")
        displacement_layout = QVBoxLayout()

        self.ux_checkbox = QCheckBox()
        self.uy_checkbox = QCheckBox()
        self.uz_checkbox = QCheckBox()
        self.u_checkbox = QCheckBox()

        # Lista de checkboxes para deslocamentos
        deslocamentos_checkboxes = [self.ux_checkbox, self.uy_checkbox, self.uz_checkbox, self.u_checkbox]

        # Conecta cada checkbox à função on_checkbox_toggled
        for checkbox in deslocamentos_checkboxes:
            checkbox.toggled.connect(lambda checked, cb=checkbox: self.on_checkbox_toggled(cb, deslocamentos_checkboxes))
        
        # Adiciona cada par de checkbox e label ao layout de tensões
        displacement_layout.addLayout(self.create_checkbox_with_label(self.ux_checkbox, "U<sub>X</sub> - Deslocamentos nodais (eixo x)"))
        displacement_layout.addLayout(self.create_checkbox_with_label(self.uy_checkbox, "U<sub>Y</sub> - Deslocamentos nodais (eixo y)"))
        displacement_layout.addLayout(self.create_checkbox_with_label(self.uz_checkbox, "U<sub>Z</sub> - Deslocamentos nodais (eixo z)"))
        displacement_layout.addLayout(self.create_checkbox_with_label(self.u_checkbox, "U - Magnitude dos deslocamentos"))

        displacement_group.setLayout(displacement_layout)
        layout.addWidget(displacement_group)

        # Grupo do tipo de análise (linear ou não-linear)
        analysis_group = QGroupBox('Tipo de análise')
        analysis_group.setStyleSheet("QGroupBox { font-size: 18px; }")
        analysis_layout = QVBoxLayout()

        self.linear_checkbox = QCheckBox('Linear-elástica')
        self.nonlinear_checkox = QCheckBox('Não-linearidade geométrica')
        self.buckling_checkbox = QCheckBox('Flambagem linear')

        self.linear_checkbox.setStyleSheet('font-size: 18px')
        self.nonlinear_checkox.setStyleSheet('font-size: 18px')
        self.buckling_checkbox.setStyleSheet('font-size: 18px')

        # Lista de checkboxes para tipo de análise
        analysis_checkboxes = [self.linear_checkbox, self.nonlinear_checkox, self.buckling_checkbox]

        # Conecta cada checkbox à função on_checkbox_toggled
        for checkbox in analysis_checkboxes:
            checkbox.toggled.connect(lambda checked, cb=checkbox: self.on_checkbox_toggled(cb, analysis_checkboxes))

        # ComboBox para seleção dos modos de flambagem
        self.modes_combobox = QComboBox()
        self.modes_combobox.addItems([f"Modo {i}" for i in range(1, num_modos + 1)])
        self.modes_combobox.setFixedHeight(30)
        self.modes_combobox.setStyleSheet('font-size: 18px')
        self.modes_combobox.setEnabled(False)

        # Conectar o checkbox da flambagem linear para habilitar/desabilitar a lista dos modos
        self.buckling_checkbox.toggled.connect(self.toggle_modes_combobox)

        # Adicionar os checkboxes ao layout corretamente
        analysis_layout.addWidget(self.linear_checkbox)
        analysis_layout.addWidget(self.nonlinear_checkox)
        analysis_layout.addWidget(self.buckling_checkbox)
        analysis_layout.addWidget(self.modes_combobox)

        # Ajustar o layout e o grupo para reduzir o tamanho
        analysis_group.setLayout(analysis_layout)
        analysis_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        analysis_group.adjustSize()
        layout.addWidget(analysis_group)

        widget.setLayout(layout)
        return widget
    
    def toggle_modes_combobox(self, checked):
        # Habilita ou desabilita o ComboBox dos modos de flambagem
        self.modes_combobox.setEnabled(checked)
            
    def deformacao(self, estrutura, tubos_ind, malha_ind, tubos_def, malha_def, desl_linear, desl_nao_linear, desl_flambagem, MT, autovalores, coords_deformadas):
        # Determinar o deslocamento nodal
        eixo = next((eixo for eixo, checkbox in [('UX', self.ux_checkbox), ('UY', self.uy_checkbox),
                                                 ('UZ', self.uz_checkbox), ('U', self.u_checkbox)] if checkbox.isChecked()), 'U')

        # Determinar o tipo de análise
        analise = next((tipo_analise for tipo_analise, checkbox in [('linear', self.linear_checkbox),
                                                                    ('não-linear', self.nonlinear_checkox),
                                                                    ('flambagem', self.buckling_checkbox)] if checkbox.isChecked()), 'linear')
        modo = self.modes_combobox.currentIndex() if analise == 'flambagem' else 0

        # Chamar a função de plotagem com os tipos selecionados e a escala
        self.stacked_widget.setCurrentIndex(1)
        plotar_deslocamentos(tubos_ind, malha_ind, tubos_def, malha_def, estrutura, desl_linear, desl_nao_linear, desl_flambagem,
                         MT, autovalores, analise, eixo, modo, coords_deformadas, self.pyvista_widget)

    def on_apply_button_clicked(self, elementos, estrutura, tubos_ind, malha_ind, tubos_def, malha_def, xp, coords,
                                esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear, desl_flambagem, 
                                autovalores, MT, pontos_int, coords_deformadas):
        # Definir a aba ativa e a escala
        current_tab = self.tabs.currentIndex()
        try:
            escala = float(self.scale_input_esforcos.text())
        except ValueError:
            escala = 1.0  # Valor padrão se a escala não for válida

        # Chamar a função adequada para a aba ativa
        if current_tab == 0:  # Aba "Esforços"
            self.esforcos(elementos, estrutura, tubos_ind, malha_ind, tubos_def, malha_def, xp,
                          coords, esforcos_lineares, esforcos_nao_lineares, MT, pontos_int, escala)
        elif current_tab == 1:  # Aba "Deformação"
            self.deformacao(estrutura, tubos_ind, malha_ind, tubos_def, malha_def, 
                            desl_linear, desl_nao_linear, desl_flambagem, MT, autovalores, coords_deformadas)

    def create_comparison_tab(self, esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear):
        widget = QWidget()
        layout = QVBoxLayout()

        # Menu de seleção para tipo de esforço/deslocamento
        self.comparison_combo = QComboBox()
        self.comparison_combo.addItems(["Fx", "Fy", "Fz", "Mx", "My", "Mz", "u", "v", "w", "θx", "θy", "θz"])
        self.comparison_combo.currentTextChanged.connect(self.update_comparison_table(esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear))
        layout.addWidget(QLabel("Selecione o tipo de esforço/deslocamento:"))
        layout.addWidget(self.comparison_combo)

        # Tabela para exibir os resultados
        self.comparison_table = QTableWidget(1, 4)
        self.comparison_table.setHorizontalHeaderLabels([
            "Tipo", "Máximo Linear", "Máximo Não-Linear", "Diferença (%)"
        ])
        layout.addWidget(self.comparison_table)

        # Conectar o sinal após a criação da tabela
        self.comparison_combo.currentTextChanged.connect(
            lambda: self.update_comparison_table(esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear)
        )

        widget.setLayout(layout)
        return widget

    def update_comparison_table(self, esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear):
        """Atualiza a tabela de comparação com base no tipo selecionado."""
        if not hasattr(self, 'comparison_table'):  # Verifica se a tabela existe
            return

        tipo = self.comparison_combo.currentText()

        # Mapear o tipo selecionado para os dados correspondentes
        data_map = {
            "Fx": (-esforcos_lineares['Fx'], -esforcos_nao_lineares['Fx']),
            "Fy": (esforcos_lineares['Fy'], esforcos_nao_lineares['Fy']),
            "Fz": (esforcos_lineares['Fz'], esforcos_nao_lineares['Fz']),
            "Mx": (esforcos_lineares['Mx'], esforcos_nao_lineares['Mx']),
            "My": (esforcos_lineares['My'], esforcos_nao_lineares['My']),
            "Mz": (esforcos_lineares['Mz'], esforcos_nao_lineares['Mz']),
            "u": (desl_linear['u'], desl_nao_linear['u']),
            "v": (desl_linear['v'], desl_nao_linear['v']),
            "w": (desl_linear['w'], desl_nao_linear['w']),
            "θx": (desl_linear['θx'], desl_nao_linear['θx']),
            "θy": (desl_linear['θy'], desl_nao_linear['θy']),
            "θz": (desl_linear['θz'], desl_nao_linear['θz']),
        }
        linear, nao_linear = data_map[tipo]

        # Calcular os valores máximos
        max_linear = np.max(linear, axis=1)
        max_nao_linear = np.max(nao_linear, axis=1)

        # Calcular a diferença percentual com verificação para evitar divisão por zero
        diff_percentual = np.where(max_linear != 0, ((max_nao_linear - max_linear) / max_linear) * 100, 0)

        # Preencher a tabela
        self.comparison_table.setRowCount(len(max_linear))
        for i in range(len(max_linear)):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(f"{tipo} (Elemento {i+1})"))
            self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{max_linear[i]:.4f}"))
            self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{max_nao_linear[i]:.4f}"))
            self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{diff_percentual[i]:.2f}%"))

        # Ajustar o tamanho das colunas
        self.comparison_table.resizeColumnsToContents()
