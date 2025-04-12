# Bibliotecas
import numpy as np
from scipy.sparse import coo_matrix


def vetor_referencia(coord):
    """
    Função para obter o vetor de orientação da seção transversal.
    """

    # Número de elementos
    elementos = coord.shape[0]

    # Inicializar vetor de referência
    ref_vector = np.zeros((elementos, 3))

    # Inverter os eixos y <-> z
    coord_invertida = np.copy(coord)
    coord_invertida[:, :, [1, 2]] = coord_invertida[:, :, [2, 1]]

    # Loop ao longo da estrutura
    for i in range(elementos):
        # Coordenadas dos nós
        L_vec = coord_invertida[i, 1] - coord_invertida[i, 0]

        # Vetor unitário ao longo do elemento
        x_ = L_vec / np.linalg.norm(L_vec)

        # Gerar vetores ortogonais y_ e z_ automaticamente
        ref_vector[i] = np.array([0, 0, 1])
        if np.allclose(ref_vector[i], np.abs(x_)):
            ref_vector[i] = np.array([1, 0, 0])

    return ref_vector


def atribuir_deslocamentos(numDOF, DOF, GLL, GLe, T, dr):
    """
    Função para atribuir os deslocamentos reduzidos (dr)
    aos graus de liberdade livres de cada elemento.

    Parâmetros:
        numDOF (int): Número total de graus de liberdade.
        DOF (int): Graus de liberdade por nó.
        GLL (np.ndarray): Array booleano indicando os graus de liberdade livres.
        GLe (np.ndarray): Array com os graus de liberdade dos elementos.
        T (np.ndarray): Matriz de transformação.
        dr (np.ndarray): Array de deslocamentos reduzidos.

    Retorna:
        np.ndarray: Array de deslocamentos locais do elemento.
    """

    # Inicializar o array de deslocamentos
    d = np.zeros((numDOF, 1))

    # Atribuir os deslocamentos aos graus de liberdade livres
    d[GLL] = dr

    # Atribuir os deslocamentos aos graus de liberdade dos elementos
    dloc = T[:, :2*DOF, :2*DOF] @ d[GLe]

    return dloc


def matriz_esparsa(elementos, ke, DOF, numDOF, GLe):
    """
    Monta a matriz de rigidez global esparsa a partir das matrizes
    de rigidez dos elementos.

    Parâmetros:
        elementos (int): Número de elementos.
        ke (np.ndarray): Matrizes de rigidez dos elementos (shape: (elementos, 2*DOF, 2*DOF)).
        DOF (int): Número de graus de liberdade por nó.
        numDOF (int): Número total de graus de liberdade.
        GLe (np.ndarray): Vetor de graus de liberdade dos elementos (shape: (elementos, 2*DOF)).

    Retorna:
        scipy.sparse.csc_matrix: Matriz de rigidez global esparsa.
    """
    # Inicialização das listas para armazenamento dos dados da matriz esparsa
    data, rows, cols = [], [], []

    # Montagem da matriz de rigidez global esparsa
    for e in range(elementos):
        gl = GLe[e]  # Graus de liberdade do elemento 'e'
        for i in range(2 * DOF):
            for j in range(2 * DOF):
                if ke[e, i, j] != 0:  # Verifica se o valor é não-nulo
                    rows.append(gl[i])
                    cols.append(gl[j])
                    data.append(ke[e, i, j])

    # Converter listas para matriz esparsa no formato CSC
    KG = coo_matrix((data, (rows, cols)), shape=(numDOF, numDOF)).tocsc()

    return KG


def forcas_locais(dl, ke, T, DOF, f=None):
    """
    Função para calcular as forças locais de cada elemento.

    Parâmetros:
        dl (np.ndarray): Vetor de deslocamentos locais.
        ke (np.ndarray): Matriz de rigidez do elemento.
        T (np.ndarray): Matriz de transformação.
        DOF (int): Graus de liberdade por nó.
        f (np.ndarray): Vetor de forças distribuídas.

    Retorna:
        np.ndarray: Vetor de forças locais do elemento.
    """

    if f is None:
        return ke @ dl
    else:
        return ke @ dl - T[:, :2*DOF, :2*DOF] @ f


def dados_geometricos_constitutivos(elementos, estrutura):
    """
    Calcula e retorna dados geométricos e constitutivos para elementos estruturais.

    Args:
        elementos (int): Número de elementos.
        estrutura (Estrutura): Uma instância da classe Estrutura contendo dados estruturais.

    Returns:
        tuple: Uma tupla contendo:
            - coord_ord (np.ndarray): Coordenadas ordenadas dos elementos.
            - L (np.ndarray): Comprimentos dos elementos.
            - A (np.ndarray): Área da seção transversal dos elementos.
            - Ix (np.ndarray): Momento de inércia em relação ao eixo x.
            - Iy (np.ndarray): Momento de inércia em relação ao eixo y.
            - Iz (np.ndarray): Momento de inércia em relação ao eixo z.
            - E (np.ndarray): Módulo de elasticidade.
            - nu (np.ndarray): Coeficiente de Poisson.
            - G (np.ndarray): Módulo de cisalhamento.
    """
    # Coordenadas ordenadas
    coord_ord = estrutura.coord[estrutura.conec]

    # Comprimento dos elementos
    L = np.linalg.norm(coord_ord[:, 1] - coord_ord[:, 0], axis=1)

    # Extrair dados geométricos e constitutivos
    geometrias = np.array([s['geometria'] for s in estrutura.secoes])
    dados = {k: np.array([s.get(k, 0) for s in estrutura.secoes]) for k in 
        ['base', 'altura', 'raio', 'raio_ext', 'raio_int', 
            'espessura', 'espessura_flange', 'espessura_alma', 'E', 'v', 'G']}

    # Inicializar dados geométricos
    A = np.zeros(elementos)
    Ix = np.zeros(elementos)
    Iy = np.zeros(elementos)
    Iz = np.zeros(elementos)
    
    # Máscaras para cada tipo de seção
    ret_mask = geometrias == 'retangular'
    caixa_mask = geometrias == 'caixa'
    circ_mask = geometrias == 'circular'
    tub_mask = geometrias == 'tubular'
    i_mask = geometrias == 'I'
    t_mask = geometrias == 'T'

    # Seções retangulares
    if np.any(ret_mask):
        b, h = dados['base'][ret_mask], dados['altura'][ret_mask]
        A[ret_mask] = b * h
        
        hx, hy = np.minimum(b, h), np.maximum(b, h)
        Ix[ret_mask] = hx**3 * hy * (1/3 - 0.21 * hx/hy * (1 - 1/12 * (hx/hy)**4))
        Iy[ret_mask] = h * b**3 / 12
        Iz[ret_mask] = b * h**3 / 12

    # Seções retangulares vazadas (caixas)
    if np.any(caixa_mask):
        b, h, t = dados['base'][caixa_mask], dados['altura'][caixa_mask], dados['espessura'][caixa_mask]
        A[caixa_mask] = b * h - (b - 2*t) * (h - 2*t)
        
        denom = h*t + b*t - 2*t**2
        Ix[caixa_mask] = 2 * t**2 * (b - 2)**2 * (h - t)**2 / denom
        Iy[caixa_mask] = (b**3 * h - (b - 2*t)**3 * (h - 2*t)) / 12
        Iz[caixa_mask] = (b * h**3 - (b - 2*t) * (h - 2*t)**3) / 12

    # Seções circulares
    if np.any(circ_mask):
        r = dados['raio'][circ_mask]
        A[circ_mask] = np.pi * r**2
        Ix[circ_mask] = Iy[circ_mask] = Iz[circ_mask] = np.pi * r**4 / 4
        Ix[circ_mask] *= 2

    # Seções circulares vazadas (tubulares)
    if np.any(tub_mask):
        re, ri = dados['raio_ext'][tub_mask], dados['raio_int'][tub_mask]
        A[tub_mask] = np.pi * (re**2 - ri**2)
        Ix[tub_mask] = Iy[tub_mask] = Iz[tub_mask] = np.pi * (re**4 - ri**4) / 4
        Ix[tub_mask] *= 2

    # Seções I
    if np.any(i_mask):
        b, h, tf, tw = (dados['base'][i_mask], dados['altura'][i_mask], 
                        dados['espessura_flange'][i_mask], dados['espessura_alma'][i_mask])
        A[i_mask] = 2 * b * tf + (h - 2*tf) * tw
        Ix[i_mask] = (2 * b * tw + (h - tf) * tw**3) / 3
        Iy[i_mask] = ((h - 2*tf) * tw**3 + 2 * tf * b**3) / 12
        Iz[i_mask] = (b * h**3 - (b - tw) * (h - 2*tf)**3) / 12

    # Seções T
    if np.any(t_mask):
        b, h, tf, tw = (dados['base'][t_mask], dados['altura'][t_mask], 
                        dados['espessura_flange'][t_mask], dados['espessura_alma'][t_mask])
        A[t_mask] = b * tf + (h - tf) * tw
        
        # Cálculo do centroide
        numerador = h**2 * tw + tf**2 * (b - tw)
        denominador = 2 * (b * tf + (h - tf) * tw)
        yc = h - (numerador / denominador)
        
        Ix[t_mask] = (b * tf**3 + (h - tf/2) * tw**3) / 3
        Iy[t_mask] = ((h - tf) * tw**3 + b**3 * tf) / 12
        Iz[t_mask] = (tw * yc**3 + b * (h - yc)**3 - (b - tw) * (h - yc - tf)**3) / 3

    # Propriedades constitutivas
    E = dados['E']
    nu = dados['v']
    G = dados['G']

    # Dicionário para armazenar todas as propriedades
    propriedades = {
        'L': L, 'A': A,
        'Ix': Ix, 'Iy': Iy, 'Iz': Iz,
        'E': E, 'nu': nu, 'G': G,
        'b': dados['base'], 'h': dados['altura'],
        'raio': dados['raio'], 'raio_ext': dados['raio_ext'], 'raio_int': dados['raio_int'],
        't': dados['espessura'], 'tf': dados['espessura_flange'], 'tw': dados['espessura_alma']
    }

    return coord_ord, propriedades


def expandir_dados(elementos, modelo, dr, fr=None, f=None):
    """
    Expandir os deslocamentos e esforços para 6 graus de liberdade.

    Parâmetros:
        elementos (int): Número de elementos.
        modelo (str): Tipo de modelo estrutural ('viga' ou 'treliça').
        dr (np.ndarray): Deslocamentos reduzidos.
        fr (np.ndarray, optional): Esforços reduzidos.
        f (np.ndarray, optional): Forças nodais.

    Retorna:
        dr_exp (np.ndarray): Deslocamentos expandidos.
        fr_exp (np.ndarray): Esforços expandidos.
        f_exp (np.ndarray): Forças nodais expandidas.
    """
    if modelo == 'viga':
        return dr, fr, f

    # Expandir os deslocamentos e esforços para 6 graus de liberdade
    idx = np.array([0, 1, 2, 6, 7, 8])

    dr_exp = np.zeros((elementos, 12, 1))
    dr_exp[:, idx, :] = dr

    fr_exp = np.zeros((elementos, 12, 1))
    if fr is not None:
        fr_exp[:, idx, :] = fr

    f_exp = np.zeros((elementos, 12, 1))
    if f is not None:
        f_exp[:, idx, :] = f

    return dr_exp, fr_exp, f_exp


def condensacao_estatica(modelo, k, f, gl_mantidos=None, n=12):
    """
    Condensação estática vetorizada para múltiplas matrizes de rigidez.

    Parâmetros:
        K (np.array): Matrizes de rigidez com shape (elementos, 12, 12).
        F (np.array): Vetores de forças com shape (elementos, 12, 1). Opcional.
        gl_mantidos (list): Graus de liberdade mantidos após a condensação.

    Retorna:
        K_cond (np.array): Matrizes de rigidez condensadas.
        F_cond (np.array): Vetores de forças condensados.
    """
    # Desconsiderar em caso de elementos de viga
    if modelo == 'viga':
        return k, f

    # Graus de liberdade livres para estruturas treliçadas
    if gl_mantidos is None:
        gl_mantidos = [0, 1, 2, 6, 7, 8]

    # Graus de liberdade eliminados
    gl_eliminados = np.array([i for i in range(n) if i not in gl_mantidos])

    # Partição das matrizes de rigidez
    k_mm = k[:, gl_mantidos][:, :, gl_mantidos]
    k_me = k[:, gl_mantidos][:, :, gl_eliminados]
    k_em = k[:, gl_eliminados][:, :, gl_mantidos]
    k_ee = k[:, gl_eliminados][:, :, gl_eliminados]

    # Adicionar pequena perturbação aos graus de liberdade eliminados
    k_ee[:, ] += 1e-8 * np.eye(k_ee.shape[1])

    # Calcular a inversa de k_ee para todos os elementos
    k_ee_inv = np.linalg.inv(k_ee)

    # Condensação das matrizes de rigidez
    k_cond = k_mm - k_me @ k_ee_inv @ k_em

    if f is None:
        return k_cond, f
    else:
        # Partição das forças globais
        f_a = f[:, gl_mantidos, :]
        f_b = f[:, gl_eliminados, :]

        # Condensação dos vetores de forças
        f_cond = f_a - k_me @ k_ee_inv @ f_b

        return k_cond, f_cond


def matriz_transformacao(elementos, coords, propriedades):
    """
    Calcula as matrizes de transformação locais e globais para múltiplos elementos,
    além de obter a deformação relativa de cada eixo local x.

    Parâmetros:
    elementos : int
        Número de elementos.
    coords : array_like
        Coordenadas dos nós dos elementos (shape: (elementos, 2, 3)).
    L : array_like
        Comprimentos dos elementos.

    Retorna:
    T : array_like
        Matriz de transformação global (shape: (elementos, 12, 12)).
    MT : array_like
        Matriz de transformação local (shape: (elementos, 3, 3)).
    ε : array_like
        Deformação relativa de cada eixo local x (shape: (elementos, 3)).
    """
    ε = np.zeros((elementos, 3))
    MT = np.zeros((elementos, 3, 3))
    T = np.zeros((elementos, 12, 12))

    # Comprimentos dos elementos
    L = propriedades['L']
    
    for i in range(elementos):
        # Vetor unitário ao longo do eixo local x
        x_ = (coords[i, 1] - coords[i, 0]) / L[i]

        # Deformação relativa de cada coordenada
        ε[i] = x_

        # Definir um vetor arbitrário (mas não paralelo a x_) para calcular o vetor ortogonal
        if np.allclose(np.abs(x_), [0, 0, 1]):
            y_ = np.array([0, 1, 0], dtype=float)
        else:
            y_ = np.cross([0, 0, 1], x_)
        y_ /= np.linalg.norm(y_)

        # Calcular o vetor ortogonal z_
        z_ = np.cross(x_, y_)
        z_ /= np.linalg.norm(z_)

        # Matriz de transformação do elemento
        MT[i] = np.array((x_, y_, z_))

        # Matriz de transformação global
        T[i] = np.kron(np.eye(4), MT[i])

    return T, MT, ε


def matriz_transformacao_deformada(de, elementos, coord, MT):
    """
    Cálculo da matriz de transformação da estrutura deformada.

    Args:
        de : array_like
            Vetor de deslocamentos globais (shape: (elementos, 12)).
        elementos : int
            Número de elementos.
        coord : array_like
            Coordenadas dos nós da estrutura (shape: (nós, 3)).
        MT : array_like
            Matriz de transformação local (shape: (elementos, 3, 3)).

    Returns:
        Td : array_like
            Matriz de transformação global deformada (shape: (elementos, 12, 12)).
        ε : array_like
            Deformação relativa de cada eixo local x (shape: (elementos, 3)).
    """
    ε = np.zeros((elementos, 3))
    Td = np.zeros((elementos, 12, 12))

    # Obter as coordenadas locais inicial e final da estrutura, cli e clf
    cli = np.einsum('ijk,ik->ij', MT, coord[:, 0])
    clf = np.einsum('ijk,ik->ij', MT, coord[:, 1])

    # Pré-calcula os vetores x_ para todos os elementos
    for i in range(elementos):
        # Coordenadas dos nós do elemento (indeformado)
        x1 = cli[i]
        x2 = clf[i]

        # Deslocamentos dos nós do elemento
        u1 = de[i][:3].flatten()
        u2 = de[i][6:9].flatten()

        # Coordenadas dos nós do elemento (deformado)
        x1d = x1 + u1
        x2d = x2 + u2

        # Vetor x_ (direção do eixo local x)
        x_ = normalize(x2d - x1d)
        ε[i] = x_

        # Vetor y_ (direção do eixo local y)
        if np.allclose(np.abs(x_), [0, 0, 1]):
            y_ = np.array([0, 1, 0], dtype=float)
        else:
            y_ = np.cross([0, 0, 1], x_)
        y_ = normalize(y_)

        # Calcular o vetor ortogonal z_
        z_ = np.cross(x_, y_)
        z_ = normalize(z_)

        # Matriz de transformação do elemento
        MTd = np.array((x_, y_, z_))

        # Expande para matriz 12x12
        Td[i] = np.kron(np.eye(4), MTd)

        # Verifica se MTd é uma matriz de rotação válida
        assert np.allclose(np.dot(MTd, MTd.T), np.eye(3)), "MTd não é ortogonal"
        assert np.isclose(np.linalg.det(MTd), 1), "Determinante de MTd não é 1"

    return Td, ε


def normalize(v):
    """
    Normaliza um vetor v.

    Args:
        v (array_like): Vetor a ser normalizado.

    Returns:
        array_like: Vetor normalizado.

    Raises:
        ValueError: Se o vetor for nulo ou muito pequeno para normalização.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:  # Evita divisão por zero
        raise ValueError("Vetor nulo ou muito pequeno para normalização")
    return v / norm


def coordenadas_deformadas(coords, dl, dnl, d_flamb, MT):
    """
    Retorna as coordenadas deformadas para análises linear, não-linear e flambagem.

    Args:
        coords (np.ndarray): Coordenadas originais dos elementos.
        dl (np.ndarray): Deslocamentos lineares (elementos, 12, 1).
        dnl (np.ndarray): Deslocamentos não lineares (elementos, 12, 1).
        d_flamb (np.ndarray): Modos de flambagem (modos, elementos, 12, 1).
        MT (np.ndarray): Matrizes de transformação (elementos, 3, 3).
        tipo (str): Tipo de análise ('linear', 'não-linear', 'flambagem').
        escala (float): Fator de escala aplicado aos deslocamentos.

    Returns:
        dict: {'linear': ..., 'não-linear': ..., 'flambagem': ...}
    """
    def deformar(d, MT, coords):
        # Extrai deslocamentos dos dois nós
        dloc = np.stack((d[:, :3], d[:, 6:9]), axis=1).squeeze()
        dglob = np.einsum('eji,enj->eni', MT, dloc, optimize=True)

        # Verifica se dglob é válido (não nulo)
        if np.size(dglob) == 0 or not np.any(dglob):
            d_max = 1
        else:
            d_max = np.max(np.abs(dglob))

        magnitude = max(1, 1 / d_max)               # Fator de escala unitário
        cdef = coords + magnitude * dglob
        cdef[:, :, [1, 2]] = cdef[:, :, [2, 1]]     # Trocar eixos y <-> z
        return cdef

    # Coordenadas deformadas linear e não linear
    coord_linear = deformar(dl, MT, coords)
    coord_nao_linear = deformar(dnl, MT, coords)

    # Inicializar coordenadas de flambagem
    elementos = dl.shape[0]
    num_modos = d_flamb.shape[0]
    coord_flambagem = np.empty((elementos, num_modos, 2, 3))

    # Calcular coordenadas de flambagem
    for idx in range(num_modos):
        coord_flambagem[:, idx] = deformar(d_flamb[idx], MT, coords)

    return {
        'linear': coord_linear,
        'não-linear': coord_nao_linear,
        'flambagem': coord_flambagem
    }