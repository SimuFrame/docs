# Importações de bibliotecas padrão
import os
import sys
import importlib

# Importações de bibliotecas de terceiros
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse.linalg import spsolve

# Importações de bibliotecas de interface gráfica (GUI)
from PySide6.QtWidgets import QApplication
from pyvistaqt import BackgroundPlotter
from GUI.main_window import MainWindow
from GUI.StartupWindow import SimuFrameWindow

# Importações de bibliotecas autorais/do projeto
from EstabilidadeEstrutural import analise_estabilidade
from MontagemForças import carga_nodal_dist, vetor_forcas_globais, forcas_internas
from MontagemMatrizElástica import graus_liberdade, matriz_elastica_analitica
from NewtonRaphson import newton_raphson
from Utilitários import (
    atribuir_deslocamentos, dados_geometricos_constitutivos, expandir_dados, forcas_locais, forcas_externas,
    vetor_referencia, matriz_transformacao, matriz_transformacao_deformada, coordenadas_deformadas
)
from Visualização import malha_indeformada, malha_deformada, plot_estrutura_matplotlib, plot_estrutura_pyvista

# Limpa a tela de forma compat vel com diferentes sistemas operacionais
os.system('cls' if os.name == 'nt' else 'clear')

# Configurar o estilo do Seaborn
sns.set_theme(style="darkgrid", palette="colorblind", font_scale=1.2)

def carregar_estrutura(analise, tipo, subtipo, n, plot):
    global estrutura, P, q, M, coords, propriedades, ref_vector, tubos_ind, secao_ind

    try:
        # Importar dados da estrutura
        modulo_name = f"Data.{tipo}"
        modulo = importlib.import_module(modulo_name)

        # Executar o exemplo
        exemplo_func = getattr(modulo, "exemplo")
        estrutura = exemplo_func(analise, subtipo + 1, n)

        # Definir parâmetros da estrutura
        nos = len(estrutura.coord)              # Número de nós
        elementos = estrutura.num_elementos     # Número de elementos

        # Obter as forças externas (concentrada, distribuída e momento)
        P, q, M = forcas_externas(elementos, nos, estrutura)

        # Obter os dados geométricos e constitutivos da seção
        coords, propriedades = dados_geometricos_constitutivos(elementos, estrutura)

        # Vetor de referência para plotagem da seção transversal dos elementos
        ref_vector = vetor_referencia(coords)

        # Criar malha da estrutura indeformada
        tubos_ind, secao_ind = malha_indeformada(coords, estrutura, ref_vector)

        # Plotar a estrutura, caso seja solicitado
        if plot:
            # Inicializar elementos de plotagem (PyVista e Matplotlib)
            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            ax = fig.add_subplot(111, projection='3d')
            plotter = BackgroundPlotter()

            # Calcular a magnitude dos esforços para a plotagem
            magnitude = 1 / max(map(np.max, (np.abs(P), np.abs(q), np.abs(M))))

            # Obter a geometria inicial (sem subdivisão)
            coord_inicial = estrutura.coord_original
            conec_inicial = estrutura.conec_original

            # Definir as cargas distribuídas inicias (sem subdivisão)
            q_inicial = np.zeros((len(conec_inicial), 2, 3), dtype=float)
            for elemento, qi, qf in estrutura.cargas_iniciais:
                q_inicial[elemento] = [qi, qf]

            # Plotar a estrutura indeformada (PyVista)
            plot_estrutura_pyvista(tubos_ind, secao_ind, plotter, estrutura)

            # Plotar a estrutura indeformada (matplotlib)
            plot_estrutura_matplotlib(ax, estrutura, coord_inicial, conec_inicial, q_inicial, P, M, magnitude, transparencia=0.7, legenda=True)
    
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Erro ao carregar a estrutura: {e}")
        return None

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    janela = SimuFrameWindow()
    janela.estruturaSelecionada.connect(carregar_estrutura)
    janela.show()
    app.exec()

# Definir parâmetros da estrutura
nome_estrutura = estrutura.nome         # Nome da estrutura
modelo = estrutura.modelo               # Modelo estrutural (viga ou treliça)
nos = len(estrutura.coord)              # Número de nós
elementos = estrutura.num_elementos     # Número de elementos
conectividade = estrutura.conec         # Matriz de conectividade

# Ordenar conectividades em ordem de ocorrência
conec_ordenada = np.unique(estrutura.conec.flatten(), return_index=True)[0]

# Determinar graus de liberdade por nó
DOF = 6 if modelo == 'viga' else 3

# Associar graus de liberdade à estrutura
GLe, numDOF, resDOF = graus_liberdade(elementos, nos, estrutura, DOF)

# Obter a matriz de transformação
T, MT, ε = matriz_transformacao(elementos, coords, propriedades)

# Obtenção das cargas nodais equivalentes devido às cargas distribuídas
fe = carga_nodal_dist(elementos, q, propriedades)

# Obtenção das matrizes de rigidez da estrutura e dos elementos
Ke, ke, f = matriz_elastica_analitica(modelo, elementos, propriedades, numDOF, DOF, GLe, T, fe)

# Obtenção das forças globais da estrutura
Fe = vetor_forcas_globais(modelo, P, M, f, GLe, numDOF, DOF, conec_ordenada)

# Obtendo os índices das condições de contorno
GLL = np.ones(numDOF, dtype=bool)
GLL[resDOF] = False

# Aplicando as condições de contorno à matriz de rigidez e ao vetor de forças
KE = Ke[GLL][:, GLL]
F = Fe[GLL]

# Resolução do sistema de equações para os deslocamentos globais
d = spsolve(KE, F).reshape(-1, 1)

# Vetor dos deslocamentos lineares, {dl}
dl = atribuir_deslocamentos(numDOF, DOF, GLL, GLe, T, d)

# Vetor das forças lineares, {fl}
fl = forcas_locais(dl, ke, T, DOF, f)

# Deslocamentos e esforços expandidos para 6 graus de liberdade (treliça)
dl, fl, f = expandir_dados(elementos, modelo, dl, fl, f)

# Esforços internos no elemento
fl, Fx, Fy, Fz, Mx, My, Mz = forcas_internas(elementos, f, fl, T, linear=True)

# Obtenção dos dados da análise de estabilidade
num_modos, autovalores, d_flamb = analise_estabilidade(modelo, elementos, propriedades, numDOF, DOF, GLe, GLL, T, KE, fl)

# Obter os deslocamentos e esforços não-lineares
dg, dnl, fnl = newton_raphson(F, KE, elementos, estrutura, propriedades, numDOF, DOF, GLL, GLe, T)

# Número de pontos de integração dos esforços
pontos_int = 25

def calcular_esforços_plotagem(elementos, propriedades, coords, dnl, MT, n):
    """
    Calcular os esforços lineares e não lineares da estrutura, armazenando-os em variáveis próprias de plotagem, 'p' (subscrito 'l' para linear, 'nl' para não-linear)
    """

    # Obter o comprimento dos elementos
    L = propriedades['L']

    # Inicializar dicionários para armazenar os resultados
    esforcos_lineares = {key: np.zeros((elementos, n)) for key in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']}
    esforcos_nao_lineares = {key: np.zeros((elementos, n)) for key in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']}

    # Obter a matriz de transformação para a estrutura deformada (não-linear)
    Td, ε_nl = matriz_transformacao_deformada(dnl, elementos, coords, MT)

    # Obtenção dos esforços internos não-lineares
    _, Fx_nl, Fy_nl, Fz_nl, Mx_nl, My_nl, Mz_nl = forcas_internas(elementos, f, fnl, Td, linear=False)

    # Criar vetor normalizado ao longo do elemento
    dx_global = (coords[:, 1] - coords[:, 0])[:, None, :] * np.linspace(0, 1, n)[None, :, None]

    # Calcular as coordenadas dos pontos de plotagem, xp
    xp = coords[:, 0, None, :] + dx_global

    # Iterar sobre o comprimento do vão, L
    for i in range(elementos):
        # Criar um vetor infinitesimal de 0 ao comprimento do elemento com n pontos
        dx = np.linspace(0, L[i], n)

        # Contribuição das cargas distribuídas aos esforços cortantes
        Vx = q[i][0, 0] * dx + (q[i][1, 0] - q[i][0, 0]) * dx**2 / (2 * L[i])
        Vy = q[i][0, 1] * dx + (q[i][1, 1] - q[i][0, 1]) * dx**2 / (2 * L[i])
        Vz = q[i][0, 2] * dx + (q[i][1, 2] - q[i][0, 2]) * dx**2 / (2 * L[i])

        # Contribuição das cargas distribuídas aos momentos fletores
        M_Vx = q[i][0, 0] * dx**2 / 2 + (q[i][1, 0] - q[i][0, 0]) * dx**3 / (6 * L[i])
        M_Vy = q[i][0, 1] * dx**2 / 2 + (q[i][1, 1] - q[i][0, 1]) * dx**3 / (6 * L[i])
        M_Vz = q[i][0, 2] * dx**2 / 2 + (q[i][1, 2] - q[i][0, 2]) * dx**3 / (6 * L[i])

        # Obtenção dos esforços lineares
        esforcos_lineares['Fx'][i] =  Fx[i][0] + Vx * ε[i, 0] + Vy * ε[i, 1] + Vz * ε[i, 2]
        esforcos_lineares['Fy'][i] =  Fy[i][0] - Vx * ε[i, 1] + Vy * ε[i, 0] + Vy * ε[i, 2]
        esforcos_lineares['Fz'][i] =  Fz[i][0] + Vz * ε[i, 1] + Vz * ε[i, 0] - Vx * ε[i, 2]
        esforcos_lineares['Mx'][i] =  Mx[i][0]
        esforcos_lineares['My'][i] =  My[i][0] + Fz[i][0] * dx + M_Vz * ε[i, 1] + M_Vz * ε[i, 0] + M_Vx * ε[i, 2]
        esforcos_lineares['Mz'][i] = -Mz[i][0] + Fy[i][0] * dx + M_Vx * ε[i, 1] + M_Vy * ε[i, 0] + M_Vy * ε[i, 2]

        # Obtenção dos esforços não-lineares
        esforcos_nao_lineares['Fx'][i] =  Fx_nl[i][0] + Vx * ε_nl[i, 0] + Vy * ε_nl[i, 1] + Vz * ε_nl[i, 2]
        esforcos_nao_lineares['Fy'][i] =  Fy_nl[i][0] - Vx * ε_nl[i, 1] + Vy * ε_nl[i, 0] + Vy * ε_nl[i, 2]
        esforcos_nao_lineares['Fz'][i] =  Fz_nl[i][0] + Vz * ε_nl[i, 1] + Vz * ε_nl[i, 0] - Vx * ε_nl[i, 2]
        esforcos_nao_lineares['Mx'][i] =  Mx_nl[i][0]
        esforcos_nao_lineares['My'][i] =  My_nl[i][0] + Fz_nl[i][0] * dx + M_Vz * ε_nl[i, 1] + M_Vz * ε_nl[i, 0] + M_Vx * ε_nl[i, 2]
        esforcos_nao_lineares['Mz'][i] = -Mz_nl[i][0] + Fy_nl[i][0] * dx + M_Vx * ε_nl[i, 1] + M_Vy * ε_nl[i, 0] + M_Vy * ε_nl[i, 2]

    return xp, esforcos_lineares, esforcos_nao_lineares


# Armazenar os esforços de plotagem
xp, esforcos_lineares, esforcos_nao_lineares = calcular_esforços_plotagem(elementos, propriedades, coords, dnl, MT, pontos_int)

# Obter as coordenadas deformadas
coords_deformadas = coordenadas_deformadas(coords, dl, dnl, d_flamb, MT)

# Criar a malha da estrutura deformada
tubos_def, secao_def = malha_deformada(coords_deformadas, elementos, estrutura, num_modos, ref_vector)

# Inicializar a interface gráfica
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(nome_estrutura, elementos, estrutura, tubos_ind, secao_ind, tubos_def, secao_def, num_modos,
                        esforcos_lineares, esforcos_nao_lineares, dl, dnl, d_flamb, xp, coords, MT, autovalores, pontos_int, coords_deformadas)
    window.show()
    sys.exit(app.exec())
