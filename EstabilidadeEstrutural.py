import numpy as np
from tqdm import tqdm
from scipy.linalg import eigh, qr
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, issparse
from Utilitários import atribuir_deslocamentos, expandir_dados
from MontagemMatrizGeométrica import matriz_geometrica_analitica


def metodo_subespaco(KE, KG, m, max_iter=40, tol=1e-6):
    """
    Método de subespaço para resolver o problema de autovalor generalizado (K - λ KG) ϕ = 0.

    Parâmetros:
        K: Matriz de rigidez elástica (n x n).
        KG: Matriz de rigidez geométrica (n x n).
        m: Número de autovalores desejados.
        max_iter: Número máximo de iterações.
        tol: Tolerância para convergência.

    Retorna:
        autovalores: Autovalores calculados.
        autovetores: Autovetores correspondentes.
    """
    n = KE.shape[0]

    # Converter para matrizes esparsas, se necessário
    if not issparse(KE):
        KE = csc_matrix(KE)
    if not issparse(KG):
        KG = csc_matrix(KG)

    # Inicialização com vetores aleatórios ortonormalizados
    Q, *_ = qr(np.random.rand(n, m), mode='economic')

    autovalores_antigos = np.zeros(m)

    for _ in tqdm(range(max_iter), desc='Executando análise de estabilidade estrutural'):
        # Passo 1: Resolver sistema linear K @ Y = KG @ Q
        Y = spsolve(KE, KG @ Q)

        # Passo 2: Ortogonalizar via QR
        Q, *_ = qr(Y, mode='economic')

        # Passo 3: Redução do problema ao subespaço
        KE_tilde = np.transpose(Q) @ KE @ Q
        KG_tilde = np.transpose(Q) @ KG @ Q

        # Passo 4: Resolução do problema de autovalores no subespaço reduzido
        autovalores, autovetores = eigh(KE_tilde, -KG_tilde)

        # Passo 5: Atualizar subespaço com os novos autovetores
        Q = Q @ autovetores

        # Passo 6: Verificar convergência
        if np.allclose(autovalores[:m], autovalores_antigos, rtol=tol):
            break
        autovalores_antigos = autovalores[:m].copy()

    return autovalores[:m], Q[:, :m] # type: ignore

def analise_estabilidade(elementos, modelo, numDOF, DOF, GLe, GLL, T, L, A, Ix, Ke_reduzida, fl, num_modos=5):
    """
    Análise de estabilidade estrutural (flambagem linear) com base no método de subespaço.

    Parâmetros:
        elementos (int): Número de elementos na estrutura.
        modelo (str): Modelo matemático empregado (treliça, viga, etc.).
        numDOF (int): Número de graus de liberdade.
        DOF (list): Graus de liberdade de cada nó.
        GLe (list): Vinculações de cada elemento.
        GLL (list): Graus de liberdade livres.
        T (list): Matrizes de transformação.
        L (list): Comprimentos dos elementos.
        A (list): Áreas dos elementos.
        Ix (list): Momentos de inércia dos elementos.
        Ke_reduzida (np.array): Matriz de rigidez reduzida.
        fl (list): Vetor de forças internas na configuração deformada.
        num_modos (int): Número de modos de flambagem a serem calculados (padrão: 5).

    Retorna:
        num_modos (int): Número de modos de flambagem encontrados.
        autovalores (list): Autovalores encontrados.
        d_flamb (list): Vetores de deslocamento para cada modo de flambagem.
    """
    # Calcula a matriz de rigidez geométrica global
    Kg = matriz_geometrica_analitica(elementos, modelo, numDOF, DOF, GLe, T, fl, L, A, Ix)

    # Aplicar as condições de contorno para a matriz de rigidez geométrica
    KE = Ke_reduzida
    KG = Kg[GLL][:, GLL]

    # Reduzir o intervalo de busca caso as matrizes sejam inferiores ao limite pré-estabelecido de 5
    num_modos = min(num_modos, KE.shape[0])

    # Inicializar os deslocamentos de flambagem
    d_flamb = np.zeros((num_modos, elementos, 12, 1))

    try:
        # Solução do sistema de autovalores e autovetores generalizados, [K_e] + λ * [K_g] = 0
        autovalores, autovetores = metodo_subespaco(KE, KG, m=num_modos)
    except Exception as e:
        print(f'Erro inesperado: {e}')
        return 0, [0], d_flamb

    # Verificar a norma dos autovetores
    normas = np.linalg.norm(autovetores, axis=0)
    mascara_norma = normas > 1e-6

    # Filtragem dos autovalores e autovetores
    autovalores_finais = autovalores[mascara_norma]
    autovetores_finais = autovetores[:, mascara_norma]

    # Montagem do vetor {d_flamb_reduzida}
    num_modos = len(autovalores_finais)
    d_flamb_reduzida = autovetores_finais.T.reshape(num_modos, -1, 1)

    # Montagem do vetor {d_flamb}
    for i in range(num_modos):
        df = atribuir_deslocamentos(numDOF, DOF, GLL, GLe, T, d_flamb_reduzida[i])
        d_flamb[i], *_ = expandir_dados(elementos, modelo, df)
    
    return num_modos, autovalores_finais, d_flamb
