# Bibliotecas padrão do Python
import os
import sys
import time

# Bibliotecas autorais
from MontagemMatrizElástica import graus_liberdade, matriz_elastica_analitica
from MontagemForças import carga_nodal_dist, vetor_forcas_globais, forcas_internas
from Utilitários import (
    atribuir_deslocamentos, dados_geometricos_constitutivos, expandir_dados, forcas_locais,
    vetor_referencia, matriz_transformacao, matriz_transformacao_deformada, coordenadas_deformadas
)
from Visualização import malha_indeformada, malha_deformada, plot_estrutura_matplotlib, plot_estrutura_pyvista
from EstabilidadeEstrutural import analise_estabilidade
from NewtonRaphson import newton_raphson

# Bibliotecas de terceiros
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# Interface gráfica (GUI)
from GUI.main_window import MainWindow, ValidacaoEstrutura
from PySide6.QtWidgets import QApplication
from pyvistaqt import BackgroundPlotter

# Dados de entrada:
import Data.Viga as Entrada

# Limpa a tela de forma compat vel com diferentes sistemas operacionais
os.system('cls' if os.name == 'nt' else 'clear')

# Configurar o estilo do Seaborn
sns.set_theme(style="darkgrid", palette="colorblind", font_scale=1.2)

# Configurar a exibi o para for ar n meros decimais com 8 casas decimais
np.set_printoptions(suppress=True, precision=8, floatmode="fixed")

# Acessa a estrutura de dados
estrutura = Entrada.estrutura

# Definir parâmetros da estrutura
nome_estrutura = estrutura.nome     # Nome da estrutura
modelo = estrutura.modelo           # Modelo estrutural (viga ou treliça)
nos = len(estrutura.coord)          # Número de nós
elementos = len(estrutura.conec)    # Número de elementos
conectividade = estrutura.conec     # Matriz de conectividade

# Ordenar conectividades em ordem de ocorrência
conec_ordenada = np.unique(estrutura.conec.flatten(), return_index=True)[0]

# Determinar graus de liberdade por nó
DOF = 6 if modelo == 'viga' else 3

# Associar graus de liberdade à estrutura
GLe, numDOF, resDOF = graus_liberdade(elementos, nos, estrutura, DOF)

# Inicializar vetores de carga
P = np.zeros((nos, 3), dtype=float)              # Carga concentrada por nó (kN)
q = np.zeros((elementos, 2, 3), dtype=float)     # Carga distribuída por extremidade do elemento (kN/m)
M = np.zeros((nos, 3), dtype=float)              # Momento concentrado por nó (kN.m)

# Aplicar cargas concentradas e momentos
for no, carga in estrutura.cargas_concentradas:
    P[no] = carga
for elemento, qi, qf in estrutura.cargas_distribuidas:
    q[elemento] = [qi, qf]
for no, momento in estrutura.momentos_concentrados:
    M[no] = momento

# Obter os dados geométricos e constitutivos da seção
coords, L, A, Ix, Iy, Iz, E, nu, G = dados_geometricos_constitutivos(elementos, estrutura)

# Obter a matriz de transformação
T, MT, ε = matriz_transformacao(elementos, coords, L)

# Vetor de referência para plotagem da seção transversal dos elementos
ref_vector = vetor_referencia(coords)

# Calcular a magnitude dos esforços para a plotagem
magnitude = 1 / max(map(np.max, (np.abs(P), np.abs(q), np.abs(M))))

# Obter a geometria inicial (sem subdivisão)
coord_inicial = estrutura.coord_original
conec_inicial = estrutura.conec_original

# Definir as cargas distribuídas inicias (sem subdivisão)
q_inicial = np.zeros((len(conec_inicial), 2, 3), dtype=float)
for elemento, qi, qf in estrutura.cargas_iniciais:
    q_inicial[elemento] = [qi, qf]

# Inicializar elementos de plotagem (PyVista e Matplotlib)
fig = plt.figure(figsize=(12, 9), tight_layout=True)
ax = fig.add_subplot(111, projection='3d')
plotter = BackgroundPlotter()

# Criar malha da estrutura indeformada
tubos_ind, secao_ind = malha_indeformada(coords, estrutura, ref_vector)

# Plotar a estrutura indeformada (PyVista)
plot_estrutura_pyvista(tubos_ind, secao_ind, plotter, estrutura)

# Plotar a estrutura indeformada (matplotlib)
plot_estrutura_matplotlib(ax, estrutura, coord_inicial, conec_inicial, q_inicial, P, M, magnitude, transparencia=0.7, legenda=True)

# Validação da estrutura pelo usuário
if __name__ == "__main__":
    ValidacaoEstrutura(nome_estrutura)

# Obtenção das cargas nodais equivalentes devido às cargas distribuídas
fe = carga_nodal_dist(elementos, q, L)

# Obtenção das matrizes de rigidez da estrutura e dos elementos
Ke, ke, f = matriz_elastica_analitica(modelo, elementos, numDOF, DOF, GLe, T, E, G, A, Ix, Iy, Iz, L, fe)

# Obtenção das forças globais da estrutura
Fe = vetor_forcas_globais(modelo, elementos, P, M, f, GLe, numDOF, DOF, conec_ordenada)

# Obtendo os índices das condições de contorno
GLL = np.ones(numDOF, dtype=bool)
GLL[resDOF] = False

# Aplicando as condições de contorno à matriz de rigidez e ao vetor de forças
KE = Ke[GLL][:, GLL]
F = Fe[GLL]

# Resolução do sistema de equações para os deslocamentos globais
d = spsolve(KE, F).reshape(-1, 1)

# Obtenção do vetor dos deslocamentos nodais dos elementos, {dl}
dl = atribuir_deslocamentos(numDOF, DOF, GLL, GLe, T, d)

# Vetor das forças internas
fl = forcas_locais(dl, ke, T, DOF, f)

# Obter os deslocamentos e esforços expandidos para 6 graus de liberdade (treliça)
dl, fl, f = expandir_dados(elementos, modelo, dl, fl, f)

# Obtenção dos esforços internos
fl, Fx, Fy, Fz, Mx, My, Mz = forcas_internas(elementos, f, fl, T, linear=True)

# Obtenção dos dados da análise de estabilidade
num_modos, autovalores, d_flamb = analise_estabilidade(elementos, modelo, numDOF, DOF, GLe, GLL, T, L, A, Ix, KE, fl)

# Inicia o timer
ti = time.time()

# Obter os deslocamentos e esforços não-lineares
dg, dnl, fnl = newton_raphson(F, KE, elementos, estrutura, numDOF, DOF, GLL, GLe, T, E, G, A, L, Ix, Iy, Iz)

# Finaliza o timer
tf = time.time()

# Número de pontos de integração dos esforços
pontos_int = 25

def campo_deslocamento(ξ, Le, n):
    """
    Calcula as funções de forma para um elemento de viga de Euler-Bernoulli.
    
    Parâmetros:
        ξ: array (n,)
            Coordenadas locais no elemento (0 <= ξ <= 1).
        Le: escalar
            Comprimentos dos elementos.
    
    Retorna:
        N: array (n, 6, 12)
            Funções de forma para todos os elementos e pontos.
    """

    # Inicializar a matriz de funções de forma
    N = np.zeros((n, 6, 12))

    # Funções de forma para u (deslocamento axial)
    Nu1 = 1 - ξ
    Nu2 = ξ

    # Funções de forma para v (deslocamento transversal em y)
    Nv1 = 1 - 3 * ξ**2 + 2 * ξ**3
    Nv2 = Le * (ξ - 2 * ξ**2 + ξ**3)
    Nv3 = 3 * ξ**2 - 2 * ξ**3
    Nv4 = Le * (-ξ**2 + ξ**3)

    # Funções de forma para θy (rotação em y)
    Nθ1 = (6 / Le) * (-ξ + ξ**2)
    Nθ2 = -1 + 4*ξ - 3*ξ**2
    Nθ3 = -Nθ1
    Nθ4 = 2*ξ - 3*ξ**2

    # Montagem das funções de forma
    N[:, 0, 0] = Nu1
    N[:, 0, 6] = Nu2

    N[:, 1, 1] = Nv1
    N[:, 1, 5] = Nv2
    N[:, 1, 7] = Nv3
    N[:, 1, 11] = Nv4

    N[:, 2, 2] = Nv1
    N[:, 2, 4] = -Nv2
    N[:, 2, 8] = Nv3
    N[:, 2, 10] = -Nv4

    N[:, 3, 3] = Nu1
    N[:, 3, 9] = Nu2

    N[:, 4, 2] = Nθ1
    N[:, 4, 4] = Nθ2
    N[:, 4, 8] = Nθ3
    N[:, 4, 10] = Nθ4

    N[:, 5, 1] = Nθ1
    N[:, 5, 5] = -Nθ2
    N[:, 5, 7] = Nθ3
    N[:, 5, 11] = -Nθ4

    return N

def calcular_esforços_plotagem(n):
    """
    Calcular os esforços lineares e não lineares da estrutura, armazenando-os em variáveis próprias de plotagem, 'p' (subscrito 'l' para linear, 'nl' para não-linear)
    """

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

def calcular_deslocamentos_plotagem():
    """
    Calcular os deslocamentos lineares, não-lineares e de flambagem-linear da estrutura.
    Retorna dicionários contendo os deslocamentos.
    """

    # Inicializar dicionários para armazenar os resultados
    deslocamentos_lineares = {key: np.zeros((elementos, 2)) for key in ['u', 'v', 'w', 'θx', 'θy', 'θz']}
    deslocamentos_nao_lineares = {key: np.zeros((elementos, 2)) for key in ['u', 'v', 'w', 'θx', 'θy', 'θz']}
    deslocamentos_flambagem = {key: np.zeros((num_modos, elementos, 2)) for key in ['u', 'v', 'w', 'θx', 'θy', 'θz']}

    # Obtenção dos deslocamentos nodais lineares
    deslocamentos_lineares['u'] = dl[:, [0, 6], 0] # type: ignore
    deslocamentos_lineares['v'] = dl[:, [1, 7], 0] # type: ignore
    deslocamentos_lineares['w'] = dl[:, [2, 8], 0] # type: ignore
    deslocamentos_lineares['θx'] = dl[:, [3, 9], 0] # type: ignore
    deslocamentos_lineares['θy'] = dl[:, [4, 10], 0] # type: ignore
    deslocamentos_lineares['θz'] = dl[:, [5, 11], 0] # type: ignore

    # Obtenção dos deslocamentos nodais não lineares
    deslocamentos_nao_lineares['u'] = dnl[:, [0, 6], 0] # type: ignore
    deslocamentos_nao_lineares['v'] = dnl[:, [1, 7], 0] # type: ignore
    deslocamentos_nao_lineares['w'] = dnl[:, [2, 8], 0] # type: ignore
    deslocamentos_nao_lineares['θx'] = dnl[:, [3, 9], 0] # type: ignore
    deslocamentos_nao_lineares['θy'] = dnl[:, [4, 10], 0] # type: ignore
    deslocamentos_nao_lineares['θz'] = dnl[:, [5, 11], 0] # type: ignore

    # Obtenção dos deslocamentos nodais de flambagem
    deslocamentos_flambagem['u'] = d_flamb[:, :, [0, 6], 0] # type: ignore
    deslocamentos_flambagem['v'] = d_flamb[:, :, [1, 7], 0] # type: ignore
    deslocamentos_flambagem['w'] = d_flamb[:, :, [2, 8], 0] # type: ignore
    deslocamentos_flambagem['θx'] = d_flamb[:, :, [3, 9], 0] # type: ignore
    deslocamentos_flambagem['θy'] = d_flamb[:, :, [4, 10], 0] # type: ignore
    deslocamentos_flambagem['θz'] = d_flamb[:, :, [5, 11], 0] # type: ignore

    return deslocamentos_lineares, deslocamentos_nao_lineares, deslocamentos_flambagem

# Armazenar os esforços de plotagem
xp, esforcos_lineares, esforcos_nao_lineares = calcular_esforços_plotagem(pontos_int)

# Armazenar os deslocamentos da estrutura
desl_linear, desl_nao_linear, desl_flambagem = calcular_deslocamentos_plotagem()

print(f"Tempo de execução: {tf - ti:.2f} s")

# Obter as coordenadas deformadas
coords_deformadas = coordenadas_deformadas(coords, dl, dnl, d_flamb, MT)

# Criar a malha da estrutura deformada
tubos_def, secao_def = malha_deformada(coords_deformadas, elementos, estrutura, num_modos, ref_vector)

# Inicializar a interface gráfica
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(nome_estrutura, elementos, estrutura, tubos_ind, secao_ind, tubos_def, secao_def, num_modos,
                        esforcos_lineares, esforcos_nao_lineares, desl_linear, desl_nao_linear,
                        desl_flambagem, xp, coords, MT, autovalores, pontos_int, coords_deformadas)
    window.show()
    sys.exit(app.exec())
