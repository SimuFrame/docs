# Bibliotecas
import numpy as np
from Utilitários import condensacao_estatica, matriz_esparsa


def matriz_geometrica_analitica(modelo, elementos, propriedades, numDOF, DOF, GLe, T, fl):
    """
    Calcula a matriz de rigidez geométrica analítica da estrutura.

    Args:
        elementos (int): Número de elementos.
        modelo (str): Tipo de modelo estrutural ('viga' ou 'treliça').
        numDOF (int): Número total de graus de liberdade.
        DOF (int): Número de graus de liberdade por nó.
        GLe (np.ndarray): Graus de liberdade dos elementos.
        T (np.ndarray): Matrizes de transformação.
        fl (np.ndarray): Vetor de forças locais.
        L (np.ndarray): Comprimentos dos elementos.
        A (np.ndarray): Área da seção transversal dos elementos.
        Ix (np.ndarray): Momento de inércia em relação ao eixo x.

    Returns:
        np.ndarray: Matriz de rigidez geométrica global.
    """
    # Inicialização da matriz de rigidez geométrica local, [kg]
    kg = np.zeros((elementos, 12, 12))

    # Extrair o esforço normal para todos os elementos
    Ng = fl[:, 6, 0]

    # Dados geométricos e constitutivos
    L = propriedades['L']
    A = propriedades['A']
    Ix = propriedades['Ix']

    # Calcular os termos da matriz geométrica para todos os elementos
    k1 = Ng / L
    k2 = (6 / 5) * k1
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
