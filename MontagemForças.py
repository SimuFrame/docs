# Bibliotecas
import numpy as np


def carga_nodal_dist(elementos, q, L):
    """
    Montagem do vetor de forças nodais equivalentes devido à carga distribuída.
    """

    def cload_dist(q, L):
        """ Contribuição das forças nodais devido à carga distribuída. """
        return np.array([
            q[:, 0, 0] * L / 2 + 3 * (q[:, 1, 0] - q[:, 0, 0]) * L / 20,
            q[:, 0, 1] * L / 2 + 3 * (q[:, 1, 1] - q[:, 0, 1]) * L / 20,
            q[:, 0, 2] * L / 2 + 3 * (q[:, 1, 2] - q[:, 0, 2]) * L / 20,
            q[:, 0, 0] * L / 2 + 7 * (q[:, 1, 0] - q[:, 0, 0]) * L / 20,
            q[:, 0, 1] * L / 2 + 7 * (q[:, 1, 1] - q[:, 0, 1]) * L / 20,
            q[:, 0, 2] * L / 2 + 7 * (q[:, 1, 2] - q[:, 0, 2]) * L / 20
        ])

    def mload_dist(q, L):
        """ Contribuição dos momentos nodais devido à carga distribuída. """
        return np.array([
            q[:, 0, 0] * L**2 / 12 + (q[:, 1, 0] - q[:, 0, 0]) * L**2 / 30,
            q[:, 0, 1] * L**2 / 12 + (q[:, 1, 1] - q[:, 0, 1]) * L**2 / 30,
            q[:, 0, 2] * L**2 / 12 + (q[:, 1, 2] - q[:, 0, 2]) * L**2 / 30,
            q[:, 0, 0] * L**2 / 12 + (q[:, 1, 0] - q[:, 0, 0]) * L**2 / 20,
            q[:, 0, 1] * L**2 / 12 + (q[:, 1, 1] - q[:, 0, 1]) * L**2 / 20,
            q[:, 0, 2] * L**2 / 12 + (q[:, 1, 2] - q[:, 0, 2]) * L**2 / 20
        ])

    # Criação do vetor do domínio dos elementos, {f}
    f = np.zeros((elementos, 12, 1))

    if q.any():
        # Esforços nodais devido à carga distribuída
        Pq = cload_dist(q, L)

        # Momentos nodais devido à carga distribuída
        Mq = mload_dist(q, L)

        # Contribuição das translações
        Pxi, Pyi, Pzi, Pxf, Pyf, Pzf = Pq

        # Contribuição das rotações
        # Mxi = -Mq[:, 1] * ε[:, 2] + Mq[:, 2] * ε[:, 1]
        # Myi =  Mq[:, 0] * ε[:, 2] - Mq[:, 2] * ε[:, 0]
        # Mzi = -Mq[:, 0] * ε[:, 1] + Mq[:, 1] * ε[:, 0]
        # Mxf = -Mq[:, 4] * ε[:, 2] + Mq[:, 5] * ε[:, 1]
        # Myf =  Mq[:, 3] * ε[:, 2] - Mq[:, 5] * ε[:, 0]
        # Mzf = -Mq[:, 3] * ε[:, 1] + Mq[:, 4] * ε[:, 0]

        # Contribuição dos momentos
        Mxi, Myi, Mzi, Mxf, Myf, Mzf = Mq

        # Atribuição das forças e momentos ao vetor de forças nodais
        f[:, [0, 1, 2], 0] = np.stack([Pxi, Pyi, Pzi], axis=1)
        f[:, [6, 7, 8], 0] = np.stack([Pxf, Pyf, Pzf], axis=1)
        f[:, [3, 4, 5], 0] = np.stack([Mzi-Myi, Mxi-Mzi, Myi-Mxi], axis=1)
        f[:, [9, 10, 11], 0] = np.stack([Myf-Mzf, Mzf-Mxf, Mxf-Myf], axis=1)

        return f

    else:
        return f


def vetor_forcas_globais(modelo, elementos, P, M, f, GLe, numDOF, DOF, conec):
    """
    Montagem do vetor de forças globais, {F}.

    Parâmetros:
        P (np.array): Vetor de forças concentradas.
        M (np.array): Vetor de momentos concentrados.
        f (np.array): Vetor de forças distribuídas.
        GLe (np.array): Vetor de graus de liberdade dos elementos.
        DOF (int): Número de graus de liberdade por nó.

    Retorna:
        F (np.array): Vetor de forças globais com shape (nós * DOF, 1).
    """

    # Número total de graus de liberdade

    # Criação do vetor dos esforços globais, {F}
    F = np.zeros((numDOF, 1))

    # Montagem do vetor {F}
    for i, no in enumerate(conec):
        # Adicionar esforços concentrados (Fx, Fy, Fz)
        Fx, Fy, Fz = P[no]  # Acessar as cargas do nó atual
        F[DOF * no:DOF * no + 3] += np.array([[Fx], [Fy], [Fz]])

        if modelo == 'viga':
            # Adicionar momentos concentrados (Mx, My, Mz)
            Mx, My, Mz = M[no]  # Acessar as cargas do nó atual
            F[DOF * no + 3:DOF * no + 6] += np.array([[-Mx], [My], [-Mz]])

    # Adicionar as cargas distribuídas à {F}
    np.add.at(F, GLe, f)

    return F


def forcas_internas(elementos, f, fl, T, linear):
    """
    Atribuir os esforços internos nos elementos da estrutura.

    Args:
        elementos (int): Número de elementos da estrutura.
        f (np.array): Vetor de forças distribuídas.
        fl (np.array): Vetor de forças locais.
        T (np.array): Matriz de transformação.
        linear (bool): Se a análise é linear ou não.

    Returns:
        fe (np.array): Vetor de esforços internos.
        Fx (np.array): Esforço interno em x.
        Fy (np.array): Esforço interno em y.
        Fz (np.array): Esforço interno em z.
        Mx (np.array): Momento interno em x.
        My (np.array): Momento interno em y.
        Mz (np.array): Momento interno em z.
    """

    # Valores dos esforços no início e fim de cada elemento
    Fx = np.zeros((elementos, 2))
    Fy = np.zeros((elementos, 2))
    Fz = np.zeros((elementos, 2))
    Mx = np.zeros((elementos, 2))
    My = np.zeros((elementos, 2))
    Mz = np.zeros((elementos, 2))

    # Esforço interno no elemento
    if linear:
        fe = fl
    else:
        fe = T @ fl - T @ f

    # Atribuições dos esforços para plotagem
    Fx = fe[:, [0, 6], 0]
    Fy = fe[:, [1, 7], 0]
    Fz = fe[:, [2, 8], 0]
    Mx = fe[:, [3, 9], 0]
    My = fe[:, [4, 10], 0]
    Mz = fe[:, [5, 11], 0]

    return fe, Fx, Fy, Fz, Mx, My, Mz
