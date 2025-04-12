# Bibliotecas
import numpy as np
from Utilitários import expandir_dados, atribuir_deslocamentos, condensacao_estatica, matriz_esparsa


def matrizes_cinematicas(elementos, ξ, A, L, Ix, de, NLG):
    """
    Calcula as matrizes cinemáticas linear e não-linear para múltiplos elementos.

    Parâmetros:
        ξ: Ponto de Gauss ao longo do comprimento (escalar).
        L: Array com os comprimentos dos elementos (shape: (num_elementos,)).
        de: Array com os deslocamentos nodais (shape: (num_elementos, 12)).
        NLG: Se True, calcula a matriz não-linear BbNL.
    """
    # Inversa do Jacobiano de x(ξ)
    J = (2 / L)[:, None, None]
    
    # Gradientes dos termos da matriz do campo de deslocamentos
    dNu1 = np.full((elementos), -0.5)
    dNu2 = np.full((elementos), 0.5)

    dNv1 = np.full((elementos), (3/4) * (ξ**2 - 1))
    dNv2 = (L/8) * (3*ξ**2 - 2*ξ - 1)
    dNv3 = -dNv1
    dNv4 = (L/8) * (3*ξ**2 + 2*ξ - 1)

    d2Nv1 = np.full((elementos), (3/2) * ξ)
    d2Nv2 = (L/4) * (3*ξ - 1)
    d2Nv3 = -d2Nv1
    d2Nv4 = (L/4) * (3*ξ + 1)

    # Criação dos vetores das matrizes cinemáticas
    dNu = np.zeros((elementos, 1, 12))
    dNv = np.zeros((elementos, 1, 12))
    dNw = np.zeros((elementos, 1, 12))
    dNθx = np.zeros((elementos, 1, 12))
    d2Nv = np.zeros((elementos, 1, 12))
    d2Nw = np.zeros((elementos, 1, 12))

    # Montagem dos vetores das matrizes cinemáticas
    dNu[:, 0, [0, 6]] = np.stack([dNu1, dNu2], axis=1)
    dNv[:, 0, [1, 5, 7, 11]] = np.stack([dNv1, dNv2, dNv3, dNv4], axis=1)
    dNw[:, 0, [2, 4, 8, 10]] = np.stack([dNv1, -dNv2, dNv3, -dNv4], axis=1)
    dNθx[:, 0, [3, 9]] = np.stack([dNu1, dNu2], axis=1)
    d2Nv[:, 0, [1, 5, 7, 11]] = np.stack([d2Nv1, d2Nv2, d2Nv3, d2Nv4], axis=1)
    d2Nw[:, 0, [2, 4, 8, 10]] = np.stack([d2Nv1, -d2Nv2, d2Nv3, -d2Nv4], axis=1)
    
    # Aplicar o Jacobiano
    dNu *= J
    dNv *= J
    dNw *= J
    dNθx *= J
    d2Nv *= J**2
    d2Nw *= J**2

    # Inicializar a matriz de gradientes
    Ge = np.zeros((elementos, 1, 12))
    BbNL = np.zeros((elementos, 1, 12))

    # Matriz não-linear de deformação-deslocamento, [BbNL]
    if NLG:
        # Matriz de gradientes
        Ge = np.concatenate([dNu, dNv, dNw, np.sqrt(Ix/A)[:, None, None] * dNθx], axis=1)

        # Matriz deformação-deslocamento não linear, [BbNL]
        BbNL = (Ge @ de).transpose(0, 2, 1) @ Ge
        BbNL = np.concatenate([BbNL, np.zeros((elementos, 3, 12))], axis=1)

    return dNu, d2Nv, d2Nw, dNθx, BbNL, Ge


def analise_elemento(de, elementos, estrutura, propriedades, NLG):
    kt = np.zeros((elementos, 12, 12))
    fe = np.zeros((elementos, 12, 1))

    # Obter os dados geométricos e constitutivos da seção
    modelo = estrutura.modelo

    # Dados geométricos e constitutivos
    L = propriedades['L']
    A = propriedades['A']
    Ix = propriedades['Ix']
    Iy = propriedades['Iy']
    Iz = propriedades['Iz']
    E = propriedades['E']
    G = propriedades['G']

    # Pontos e pesos de Gauss (comprimento)
    xg, Wgx = np.polynomial.legendre.leggauss(3)

    # Matriz constitutiva
    D = np.apply_along_axis(np.diag, 1, np.array([E * A, E * Iz, E * Iy, G * Ix]).T)

    # Expandir o vetor de deslocamentos (treliça)
    d_exp, *_ = expandir_dados(elementos, modelo, de)

    # Jacobiano do campo de deformação
    J = (L / 2)[:, None, None]

    for ξ, Wt in zip(xg, Wgx):
        # Matrizes cinemáticas
        Bu, Bbv, Bbw, Bθ, BNL, Ge = matrizes_cinematicas(elementos, ξ, A, L, Ix, d_exp, NLG)

        # Matriz deformação-deslocamento linear, [BL]
        BL = np.concatenate([Bu, Bbv, Bbw, Bθ], axis=1)

        # Matriz deformação-deslocamento total, [B]
        B = BL + BNL

        # Matriz deformação-deslocamento incremental, [Bε]
        Bε = BL + 0.5 * BNL

        # Tensões internas, [S]
        S = np.einsum('eij,ejk,ekl->eil', D, Bε, d_exp, optimize='optimal')

        # Matriz de rigidez elástica local, [ke]
        ke = np.einsum('eji,ejk,ekl->eil', B, D, B, optimize='optimal')

        # Matriz de rigidez geométrica local, [kg]
        kg = S[:, :1] * np.einsum('eji,ejk->eik', Ge, Ge, optimize='optimal')

        # Matriz de rigidez tangente local, [kt]
        kt += Wt * J * (ke + kg)

        # Vetor de forças internas, [fe]
        fe += Wt * J * np.einsum('eji,ejk->eik', B, S, optimize='optimal')

    # Aplicar condensação estática (treliças)
    kt, fe = condensacao_estatica(modelo, kt, fe)

    return kt, fe

def analise_global(d, elementos, estrutura, propriedades, numDOF, DOF, GLL, GLe, T, NLG):
    """
    Calcula as matrizes de rigidez e vetores de força internos para todos os elementos
    e monta as matrizes globais e o vetor de forças internas.

    Parameters
    ----------
    d : array_like
        Vetor de deslocamentos globais.
    elementos : int
        Número de elementos.
    estrutura : objeto
        Instância da classe Estrutura.
    numDOF : int
        Número de graus de liberdade.
    DOF : int
        Número de graus de liberdade por nó.
    GLL : array_like
        Nós globais.
    GLe : array_like
        Elementos globais.
    T : array_like
        Matriz de transformação local-global.
    E : array_like
        Módulo de elasticidade.
    nu : array_like
        Coeficiente de Poisson.
    A : array_like
        Área da seção transversal.
    L : array_like
        Comprimento dos elementos.
    Ix : array_like
        Momento de inércia.
    NLG : bool
        Se True, calcula a matriz de rigidez geométrica.

    Returns
    -------
    KG_reduzida : array_like
        Matriz de rigidez global reduzida.
    Fint : array_like
        Vetor de forças internas.
    fe : array_like
        Vetor de forças internas do elemento.
    de : array_like
        Vetor de deslocamentos locais do elemento.
    """
    # Inicialização do vetor de forças globais, {Fe}
    Fe = np.zeros((numDOF, 1))

    # Criação dos deslocamentos locais do elemento
    de = atribuir_deslocamentos(numDOF, DOF, GLL, GLe, T, d)

    # Calcula as matrizes de rigidez e vetores de força internos para todos os elementos
    ke, fe = analise_elemento(de, elementos, estrutura, propriedades, NLG)

    # Definir a matriz de transformação (condensada, caso aplicável)
    Te = T[:, :2*DOF, :2*DOF]

    # Converter para o sistema global
    ke_global = np.einsum('eji,ejk,ekl->eil', Te, ke, Te, optimize='optimal')
    fe_global = np.einsum('eji,ejk->eik', Te, fe, optimize='optimal')

    # Montar a matriz de rigidez global e o vetor de forças
    np.add.at(Fe, (GLe[:, :]), fe_global)

    # Montagem da matriz de rigidez tangente global, [KG]
    KG = matriz_esparsa(elementos, ke_global, DOF, numDOF, GLe)

    # Aplicação das condições de contorno
    KG_reduzida = KG[GLL][:, GLL]
    Fint = Fe[GLL]

    return KG_reduzida, Fint, fe, de
