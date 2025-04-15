# Bibliotecas
import numpy as np
from Utilitários import condensacao_estatica, matriz_esparsa


def graus_liberdade(elementos, nos, estrutura, DOF):
    """
    Define os graus de liberdade restritos da estrutura.

    Parâmetros:
        elementos: Lista de elementos da estrutura.
        nos: Número de nós da estrutura.
        estrutura: Objeto contendo as condições de contorno.
        DOF: Número de graus de liberdade por nó.

    Retorna:
        resDOF: Vetor com os índices dos graus de liberdade restritos.
    """
    # Número de graus de liberdade da estrutura
    numDOF = DOF * nos

    # Dicionário para mapear vínculos aos graus de liberdade restritos
    vinculos = {'viga': {
        'XSYMM': [0, 4, 5],
        'YSYMM': [1, 3, 5],
        'ZSYMM': [2, 3, 4],
        'XASYMM': [1, 2, 3],
        'YASYMM': [0, 2, 4],
        'ZASYMM': [0, 1, 5],
        'ARTICULADO': [0, 1, 2],
        'FIXOXY': [0, 1, 2, 3, 4],
        'FIXOXZ': [0, 1, 2, 3, 5],
        'FIXOYZ': [0, 1, 2, 4, 5],
        'ENGASTE': list(range(6))
    },
        'treliça': {
        'XSYMM': [0],
        'YSYMM': [1],
        'ZSYMM': [2],
        'XASYMM': [1, 2],
        'YASYMM': [0, 2],
        'ZASYMM': [0, 1],
        'ARTICULADO': [0, 1, 2],
        'ENGASTE': [0, 1, 2],
        'FIXO': [0, 1, 2]
    }}

    # Graus de liberdade restritos (com apoios)
    resDOF = []

    # Verifica o modelo da estrutura
    if estrutura.modelo == 'viga':
        _vinculos = vinculos['viga']
    elif estrutura.modelo == 'treliça':
        _vinculos = vinculos['treliça']
    else:
        raise ValueError("Estrutura inválida. Use 'treliça' ou 'viga'.")

    # Itera sobre as vinculações
    for i, vinculo in enumerate(estrutura.vinculacoes):
        idx = DOF * i  # Índice base para o nó atual

        if vinculo in _vinculos:
            resDOF.extend([idx + dof for dof in _vinculos[vinculo]])

    # Transformar em um ndarray
    resDOF = np.array(resDOF, dtype=int)

    # Criação do vetor de graus de liberdade dos elementos, {GLe}
    GLe = np.zeros((elementos, 2 * DOF), dtype=int)

    # Montagem do vetor {GLe}
    for e in range(elementos):
        conec = estrutura.conec[e]
        GLe[e] = np.hstack([DOF * conec[0] + np.arange(DOF),
                            DOF * conec[1] + np.arange(DOF)])

    return GLe, numDOF, resDOF

def matriz_elastica_analitica(modelo, elementos, propriedades, numDOF, DOF, GLe, T, f):
    """
    Calcula a matriz de rigidez elástica analítica da estrutura.

    Args:
        modelo: Tipo de modelo estrutural ('viga' ou 'treliça').
        elementos: Número de elementos na estrutura.
        numDOF: Número total de graus de liberdade.
        DOF: Número de graus de liberdade por nó.
        GLe: Graus de liberdade dos elementos.
        T: Matrizes de transformação.
        E: Módulo de elasticidade.
        G: Módulo de cisalhamento.
        A: Área da seção transversal.
        Ix, Iy, Iz: Momentos de inércia.
        L: Comprimento dos elementos.
        f: Vetor de forças.

    Returns:
        KE: Matriz de rigidez global.
        ke: Matrizes de rigidez locais.
        fe: Vetor de forças internas do elemento.
    """
    # Inicialização da matriz de rigidez elástica local
    ke = np.zeros((elementos, 12, 12))

    # Dados geométricos e constitutivos
    L = propriedades['L']
    A = propriedades['A']
    E = propriedades['E']
    G = propriedades['G']
    Ix = propriedades['Ix']
    Iy = propriedades['Iy']
    Iz = propriedades['Iz']

    # Termos da matriz de rigidez elástica
    k1 = E * A / L
    k2 = 12 * E * Iz / L**3
    k3 = 6 * E * Iz / L**2
    k4 = 4 * E * Iz / L
    k5 = 2 * E * Iz / L
    k6 = 12 * E * Iy / L**3
    k7 = 6 * E * Iy / L**2
    k8 = 4 * E * Iy / L
    k9 = 2 * E * Iy / L
    k10 = G * Ix / L

    # Preencher a matriz ke para todos os elementos
    ke[:, 0, 0] = k1
    ke[:, 0, 6] = -k1
    ke[:, 6, 0] = -k1
    ke[:, 6, 6] = k1

    ke[:, 1, 1] = k2
    ke[:, 1, 5] = k3
    ke[:, 1, 7] = -k2
    ke[:, 1, 11] = k3

    ke[:, 2, 2] = k6
    ke[:, 2, 4] = -k7
    ke[:, 2, 8] = -k6
    ke[:, 2, 10] = -k7

    ke[:, 3, 3] = k10
    ke[:, 3, 9] = -k10
    ke[:, 9, 3] = -k10
    ke[:, 9, 9] = k10

    ke[:, 4, 2] = -k7
    ke[:, 4, 4] = k8
    ke[:, 4, 8] = k7
    ke[:, 4, 10] = k9

    ke[:, 5, 1] = k3
    ke[:, 5, 5] = k4
    ke[:, 5, 7] = -k3
    ke[:, 5, 11] = k5

    ke[:, 7, 1] = -k2
    ke[:, 7, 5] = -k3
    ke[:, 7, 7] = k2
    ke[:, 7, 11] = -k3

    ke[:, 8, 2] = -k6
    ke[:, 8, 4] = k7
    ke[:, 8, 8] = k6
    ke[:, 8, 10] = k7

    ke[:, 10, 2] = -k7
    ke[:, 10, 4] = k9
    ke[:, 10, 8] = k7
    ke[:, 10, 10] = k8

    ke[:, 11, 1] = k3
    ke[:, 11, 5] = k5
    ke[:, 11, 7] = -k3
    ke[:, 11, 11] = k4

    # Realizar a condensação estática (treliças)
    ke, fe = condensacao_estatica(modelo, ke, f)

    # Definir a matriz de transformação
    Te = T[:, :2*DOF, :2*DOF]

    # Transformar para coordenadas globais
    ke_global = np.einsum('eji,ejk,ekl->eil', Te, ke, Te, optimize=True)

    # Montar a matriz de rigidez elástica global, [KE]
    KE = matriz_esparsa(elementos, ke_global, DOF, numDOF, GLe)

    return KE, ke, fe
