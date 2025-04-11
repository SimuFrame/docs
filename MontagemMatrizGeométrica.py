# Bibliotecas
import numpy as np
from Utilitários import condensacao_estatica, matriz_esparsa

# Função para montagem da matriz de rigidez geométrica global, [Kg]
def MatrizGeometricaAnaliticaCompleta(elementos, modelo, numDOF, DOF, GLe, T, fl, L, A, Ix):
    # Inicialização da matriz de rigidez geométrica global, [KG]
    KG = np.zeros((numDOF, numDOF))

    # Extrair esforços internos para todos os elementos
    Fx2 = fl[:, 6, 0]
    My1 = fl[:, 4, 0]
    Mz1 = fl[:, 5, 0]
    Mx2 = fl[:, 9, 0]
    My2 = fl[:, 10, 0]
    Mz2 = fl[:, 11, 0]

    # Calcular os termos da matriz geométrica para todos os elementos
    k1 = Fx2 / L
    k2 = (6/5) * k1
    k3 = My1 / L
    k4 = Mx2 / L
    k5 = Fx2 / 10
    k6 = My2 / L
    k7 = Mz1 / L
    k8 = Mz2 / L
    k9 = Fx2 * Ix / (A * L)
    k10 = (2 * Mz1 - Mz2) / 6
    k11 = (2 * My1 - My2) / 6
    k12 = (Mz1 + Mz2) / 6
    k13 = (My1 + My2) / 6
    k14 = 2 * Fx2 * L / 15
    k15 = Fx2 * L / 30
    k16 = Mx2 / 2
    k17 = (Mz1 - 2 * Mz2) / 6
    k18 = (My1 - 2 * My2) / 6

    # Montar a matriz de rigidez geométrica para todos os elementos
    kg = np.zeros((elementos, 12, 12))

    # Preencher a matriz kg para todos os elementos
    kg[:, 0, 0] = k1
    kg[:, 0, 6] = -k1
    kg[:, 6, 0] = -k1
    kg[:, 6, 6] = k1

    kg[:, 1, 1] = k2
    kg[:, 1, 3] = k3
    kg[:, 1, 4] = k4
    kg[:, 1, 5] = k5
    kg[:, 1, 7] = -k2
    kg[:, 1, 9] = k6
    kg[:, 1, 10] = -k4
    kg[:, 1, 11] = k5

    kg[:, 2, 2] = k2
    kg[:, 2, 3] = k7
    kg[:, 2, 4] = -k5
    kg[:, 2, 5] = k4
    kg[:, 2, 8] = -k2
    kg[:, 2, 9] = k8
    kg[:, 2, 10] = -k5
    kg[:, 2, 11] = -k4

    kg[:, 3, 1] = k3
    kg[:, 3, 2] = k7
    kg[:, 3, 3] = k9
    kg[:, 3, 4] = -k10
    kg[:, 3, 5] = k11
    kg[:, 3, 7] = -k3
    kg[:, 3, 8] = -k7
    kg[:, 3, 9] = -k9
    kg[:, 3, 10] = -k12
    kg[:, 3, 11] = k13

    kg[:, 4, 1] = k4
    kg[:, 4, 2] = -k5
    kg[:, 4, 3] = -k10
    kg[:, 4, 4] = k14
    kg[:, 4, 7] = -k4
    kg[:, 4, 8] = k5
    kg[:, 4, 9] = -k12
    kg[:, 4, 10] = -k15
    kg[:, 4, 11] = k16

    kg[:, 5, 1] = k5
    kg[:, 5, 2] = k4
    kg[:, 5, 3] = k11
    kg[:, 5, 5] = k14
    kg[:, 5, 7] = -k5
    kg[:, 5, 8] = -k4
    kg[:, 5, 9] = k13
    kg[:, 5, 10] = -k16
    kg[:, 5, 11] = -k15

    kg[:, 7, 1] = -k2
    kg[:, 7, 3] = -k3
    kg[:, 7, 4] = -k4
    kg[:, 7, 5] = -k5
    kg[:, 7, 7] = k2
    kg[:, 7, 9] = -k6
    kg[:, 7, 10] = k4
    kg[:, 7, 11] = -k5

    kg[:, 8, 2] = -k2
    kg[:, 8, 3] = -k7
    kg[:, 8, 4] = k5
    kg[:, 8, 5] = -k4
    kg[:, 8, 8] = k2
    kg[:, 8, 9] = -k8
    kg[:, 8, 10] = k5
    kg[:, 8, 11] = k4

    kg[:, 9, 1] = k6
    kg[:, 9, 2] = k8
    kg[:, 9, 3] = -k9
    kg[:, 9, 4] = -k12
    kg[:, 9, 5] = k13
    kg[:, 9, 7] = -k6
    kg[:, 9, 8] = -k8
    kg[:, 9, 9] = k9
    kg[:, 9, 10] = k17
    kg[:, 9, 11] = -k18

    kg[:, 10, 1] = -k4
    kg[:, 10, 2] = -k5
    kg[:, 10, 3] = -k12
    kg[:, 10, 4] = -k15
    kg[:, 10, 5] = -k16
    kg[:, 10, 7] = k4
    kg[:, 10, 8] = k5
    kg[:, 10, 9] = k17
    kg[:, 10, 10] = k14

    kg[:, 11, 1] = k5
    kg[:, 11, 2] = -k4
    kg[:, 11, 3] = k13
    kg[:, 11, 4] = k16
    kg[:, 11, 5] = -k15
    kg[:, 11, 7] = -k5
    kg[:, 11, 8] = k4
    kg[:, 11, 9] = -k18
    kg[:, 11, 11] = k14

    # Realizar a condensação estática (treliças)
    kg, _ = condensacao_estatica(modelo, kg, None)

    # Definir a matriz de transformação
    Te = T[:, :2*DOF, :2*DOF]

    # Transformar para coordenadas globais
    kg_global = Te.transpose(0, 2, 1) @ kg @ Te

    # Montagem da matriz de rigidez elástica global, [Ke]
    for i in range(elementos):
        idx = np.ix_(GLe[i], GLe[i])
        KG[idx] += kg_global[i]
    
    # Adicionar pequeno valor à diagonal para evitar singularidade
    KG += 1e-8 * np.eye(KG.shape[0])

    return KG

# Função para montagem da matriz de rigidez geométrica global, [Kg]
def matriz_geometrica_analitica(elementos, modelo, numDOF, DOF, GLe, T, fl, L, A, Ix):
    # Inicialização da matriz de rigidez geométrica local, [kg]
    kg = np.zeros((elementos, 12, 12))

    # Extrair o esforço normal para todos os elementos
    Ng = fl[:, 6, 0]

    # Calcular os termos da matriz geométrica para todos os elementos
    k1 = Ng / L
    k2 = (6/5) * k1
    k3 = Ng / 10
    k4 = k1 * (Ix / A)
    k5 = k1 * (2 * L**2 / 15)
    k6 = k1 * (-L**2 / 30)

    # Preencher a matriz kg para todos os elementos
    kg[:, 0, 0] = k1
    kg[:, 0, 6] = -k1
    kg[:, 6, 0] = -k1
    kg[:, 6, 6] = k1

    kg[:, 1, 1] = k2
    kg[:, 1, 5] = k3
    kg[:, 1, 7] = -k2
    kg[:, 1, 11] = k3

    kg[:, 2, 2] = k2
    kg[:, 2, 4] = -k3
    kg[:, 2, 8] = -k2
    kg[:, 2, 10] = -k3

    kg[:, 3, 3] = k4
    kg[:, 3, 9] = -k4
    kg[:, 9, 3] = -k4
    kg[:, 9, 9] = k4

    kg[:, 4, 2] = -k3
    kg[:, 4, 4] = k5
    kg[:, 4, 8] = k3
    kg[:, 4, 10] = k6

    kg[:, 5, 1] = k3
    kg[:, 5, 5] = k5
    kg[:, 5, 7] = -k3
    kg[:, 5, 11] = k6

    kg[:, 7, 1] = -k2
    kg[:, 7, 5] = -k3
    kg[:, 7, 7] = k2
    kg[:, 7, 11] = -k3

    kg[:, 8, 2] = -k2
    kg[:, 8, 4] = k3
    kg[:, 8, 8] = k2
    kg[:, 8, 10] = k3

    kg[:, 10, 2] = -k3
    kg[:, 10, 4] = k6
    kg[:, 10, 8] = k3
    kg[:, 10, 10] = k5

    kg[:, 11, 1] = k3
    kg[:, 11, 5] = k6
    kg[:, 11, 7] = -k3
    kg[:, 11, 11] = k5

    # Realizar a condensação estática (treliças)
    kg, _ = condensacao_estatica(modelo, kg, None)

    # Definir a matriz de transformação
    Te = T[:, :2*DOF, :2*DOF]

    # Transformar para o sistema global
    kg_global = np.einsum('eji,ejk,ekl->eil', Te, kg, Te, optimize=True)

    # Montar a matriz de rigidez geométrica global, [KG]
    KG = matriz_esparsa(elementos, kg_global, DOF, numDOF, GLe)

    return KG
