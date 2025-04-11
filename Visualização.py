# Bibliotecas
import numpy as np
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_apoio(ax, plotter, estrutura, widget):
    # Representar os apoios
    if widget == 'matplotlib':
        for coord, vinculo in zip(estrutura.coord, estrutura.vinculacoes):
            if vinculo == 'ENGASTE':
                marker = 's'
            elif 'FIXO' in vinculo:
                marker = 6
            else:
                marker = None
            ax.plot(coord[0], coord[2], coord[1], linewidth=2, color="dimgrey", marker=marker, ms=20)

    elif widget == 'pyvista':
        for coord, vinculo in zip(estrutura.coord, estrutura.vinculacoes):
            x, z, y = coord
            if vinculo == 'ENGASTE':
                cubo = pv.Cube(center=(x, y, z), x_length=0.2, y_length=0.2, z_length=0.2)
                plotter.add_mesh(cubo, color='gray')

            elif 'FIXO' in vinculo:
                cone = pv.Cone(center=(x, y, z-0.20), direction=(0, 0, 1), height=0.4, radius=0.2, resolution=100)
                plotter.add_mesh(cone, color='gray')
    else:
        raise ValueError("Widget inválido. Use 'matplotlib' ou 'pyvista'.")

def matriz_rotacao_malha(L_vec, ref_vector):
    """
    Função para obter a matriz de transformação para plotagem
    """

    # Vetor unitário ao longo do elemento
    x_ = L_vec / np.linalg.norm(L_vec) 

    # Calcular y_ e z_
    y_ = np.cross(ref_vector, x_)
    y_ /= np.linalg.norm(y_)
    z_ = np.cross(x_, y_)

    return np.stack((y_, z_, x_), axis=-1)


def criar_malha_elemento(coord, estrutura, ref_vector):
    """
    Função para criar a seção transversal da estrutura com o PyVista.
    Válida tanto para estrutura deformada como indeformada.
    """

    # Obter os dados geométricos e constitutivos do elemento
    coord1, coord2 = coord
    L_vec = coord2 - coord1
    rot_matrix = matriz_rotacao_malha(L_vec, ref_vector)
    dados = estrutura.secao
    secao_transversal = dados['seção']

    # Criar a seção transversal
    if secao_transversal == 'retangular':
        b, h = dados['base'], dados['altura']

        # Definir pontos para o perfil retangular
        pontos = np.array([[-b/2, -h/2, 0], [b/2, -h/2, 0], [b/2, h/2, 0], [-b/2, h/2, 0]])

        # Definir faces
        faces = np.array([[4, 0, 1, 2, 3]])

        # Criar a seção
        secao = pv.PolyData(pontos, faces)
    
    elif secao_transversal == 'caixa':
        b, h, t = dados['base'], dados['altura'], dados['espessura']

        # Definir pontos
        pontos = np.array([
            # Retângulo externo
            [-b/2, -h/2, 0],
            [ b/2, -h/2, 0],
            [ b/2,  h/2, 0],
            [-b/2,  h/2, 0],

            # Retângulo interno
            [-b/2 + t, -h/2 + t, 0],
            [ b/2 - t, -h/2 + t, 0],
            [ b/2 - t,  h/2 - t, 0],
            [-b/2 + t,  h/2 - t, 0]
        ])

        # Região entre os contornos
        triangles = np.array([
            # Triângulos conectando externo e interno
            [0, 4, 5], [0, 5, 1],  # Lado inferior
            [1, 5, 6], [1, 6, 2],  # Lado direito
            [2, 6, 7], [2, 7, 3],  # Lado superior
            [3, 7, 4], [3, 4, 0]   # Lado esquerdo
        ])

        # Definir secção
        secao = pv.PolyData(pontos, faces=np.hstack([[3, *t] for t in triangles]))

    elif secao_transversal == 'circular':
        secao = pv.Circle(radius=dados['raio'], resolution=50)

    elif secao_transversal == 'tubular':
        secao = pv.Disc(inner=dados['raio_int'], outer=dados['raio_ext'], r_res=50, c_res=50)

    elif secao_transversal == 'I':
        b, h, tf, tw = dados['base'], dados['altura'], dados['espessura_flange'], dados['espessura_alma']

        # Definir pontos para o perfil I
        pontos = np.array([
            # Mesa inferior
            [-b/2, -h/2, 0],
            [-b/2, -h/2 + tf, 0],
            [b/2, -h/2 + tf, 0],
            [b/2, -h/2, 0],
            
            # Alma
            [-tw/2, -h/2 + tf, 0],
            [-tw/2, h/2 - tf, 0],
            [tw/2, h/2 - tf, 0],
            [tw/2, -h/2 + tf, 0],
            
            # Mesa superior
            [-b/2, h/2, 0],
            [b/2, h/2, 0],
            [b/2, h/2 - tf, 0],
            [-b/2, h/2 - tf, 0]
        ])
        
        # Definir faces
        faces = np.hstack([
            [4, 0, 1, 2, 3],   # Mesa inferior
            [4, 4, 5, 6, 7],   # Alma
            [4, 8, 9, 10, 11]  # Mesa superior
        ])

        # Criar a seção
        secao = pv.PolyData(pontos, faces)
    
    elif secao_transversal == 'T':
        b, h, tf, tw = dados['base'], dados['altura'], dados['espessura_flange'], dados['espessura_alma']

        # Definir pontos para o perfil T
        pontos = np.array([
            # Mesa superior
            [-b/2, h/2, 0],
            [b/2, h/2, 0],
            [b/2, h/2 - tf, 0],
            [-b/2, h/2 - tf, 0],
            
            # Alma
            [-tw/2, -h/2, 0],
            [-tw/2, h/2 - tf, 0],
            [tw/2, h/2 - tf, 0],
            [tw/2, -h/2, 0]
        ])

        # Definir faces
        faces = np.hstack([
            [4, 0, 1, 2, 3],   # Mesa superior
            [4, 4, 5, 6, 7]    # Alma
        ])

        # Criar a seção
        secao = pv.PolyData(pontos, faces)

    else:
        raise ValueError(f"Tipo de seção '{secao_transversal}' não suportado.")

    if secao is not None:
        # Aplicar a matriz de transformação
        secao.points = secao.points @ rot_matrix.T + coord1 # type: ignore
        return secao.extrude(L_vec, capping=True)
    else:
        raise ValueError("Seção não foi criada: seção '{secao_transversal}' não suportado.")


def criar_malhas_em_paralelo(i, pontos, estrutura, ref_vector, deformada=False, plotar_secao=True):
    """
    Função que encapsula a criação da malha e a atribuição dos escalares.
    """
    # Inicializar dicionários para malha e tubo
    malha = {'linear': None, 'não-linear': None, 'flambagem': []}
    tubo = {'linear': None, 'não-linear': None, 'flambagem': []}

    # Criar a malha do elemento (deformada ou indeformada)
    if deformada:
        # Processar cada tipo de análise
        for analise in pontos:
            coords_list = pontos[analise][i] if analise == 'flambagem' else [pontos[analise][i]]
            for modo, coords in enumerate(coords_list):
                if plotar_secao:
                    malha_atual = criar_malha_elemento(coords, estrutura, ref_vector[i])
                else:
                    malha_atual = None

                tubo_atual = pv.lines_from_points(coords).tube(radius=0.02)

                # Armazenar resultados
                if analise == 'flambagem':
                    malha['flambagem'].append(malha_atual)
                    tubo['flambagem'].append(tubo_atual)
                else:
                    malha[analise] = malha_atual
                    tubo[analise] = tubo_atual
    else:
        # Processar caso não deformado
        coords = pontos[i]
        if plotar_secao:
            malha = criar_malha_elemento(coords, estrutura, ref_vector[i])
        else:
            malha = None
        tubo = pv.lines_from_points(coords).tube(radius=0.02)
    return tubo, malha


def plot_cargas_distribuidas(ax, coord, conec, q, magnitude):
    for i, (no_inicial, no_final) in enumerate(conec):
        pt1 = coord[no_inicial]
        pt2 = coord[no_final]

        # Verificar se há carga distribuída no elemento
        if np.any(q[i] != 0):
            for k in range(3):  # Iterar sobre x, y, z
                # Atribuir as cargas por eixo separadamente
                q1 = np.zeros(3)
                q2 = np.zeros(3)
                q1[k], q2[k] = q[i, 0, k], q[i, 1, k]

                # Verificar se há carga não nula para o eixo analisado
                if q1[k] != 0 or q2[k] != 0:
                    # Dividir o elemento em segmentos para representar a carga distribuída
                    num_flechas = 5  # Número de flechas
                    segmentos = np.linspace(0, 1, num_flechas + 1)

                    # Calcular as posições intermediárias e as cargas interpoladas
                    pontos_intermediarios = pt1 + np.outer(segmentos, (pt2 - pt1))
                    cargas = np.outer(segmentos, (q2 - q1)) + q1
                    cargas *= magnitude  # Aplicar a magnitude da carga

                    # Inicializar listas para armazenar extremidades das flechas
                    x_arrows, y_arrows, z_arrows = [], [], []

                    for ponto, carga in zip(pontos_intermediarios, cargas):
                        # Coordenadas da ponta da flecha (extremidade)
                        x, y, z = ponto - carga
                        x_arrows.append(x)
                        y_arrows.append(y)
                        z_arrows.append(z)

                        # Desenhar a flecha no gráfico
                        ax.quiver(x, z, y, carga[0], carga[2], carga[1],
                                  color='red', arrow_length_ratio=0.2, linewidth=1.5)

                    # Conectar as extremidades das flechas com uma linha
                    ax.plot(x_arrows, z_arrows, y_arrows, color="red", linestyle='-')

                    # Exibir valores das cargas distribuídas
                    ponto_medio = (pt1 + pt2) / 2
                    if np.array_equal(q1, q2):
                        ax.text(ponto_medio[0] - cargas[0, 0],
                                ponto_medio[2] - cargas[0, 2] + 0.1,
                                ponto_medio[1] - cargas[0, 1],
                                f'{q1[k]:.2f} kN/m', color='firebrick',
                                fontsize=16, fontweight='bold', ha='center', va='bottom')
                    else:
                        ax.text(pt1[0] - cargas[0, 0],
                                pt1[2] - cargas[0, 2] + 0.1,
                                pt1[1] - cargas[0, 1],
                                f'{q1[k]:.2f} kN/m', color='firebrick',
                                fontsize=16, ha='center', va='bottom')
                        ax.text(pt2[0] - cargas[-1, 0],
                                pt2[2] - cargas[-1, 2] + 0.1,
                                pt2[1] - cargas[-1, 1],
                                f'{q2[k]:.2f} kN/m', color='firebrick',
                                fontsize=16, fontweight='bold', ha='center', va='bottom')


def plot_cargas_concentradas(ax, estrutura, P, magnitude):
    """
    Plota as cargas concentradas nos nós da estrutura.
    """
    for i, (x, y, z) in enumerate(estrutura.coord):
        if np.any(P[i] != 0):  # Verifica se há cargas não nulas
            for j, carga in enumerate(P[i]):
                if carga != 0:
                    p_vetor = np.zeros(3)
                    p_vetor[j] = carga

                    # Calcula as componentes do vetor da flecha
                    u, v, w = magnitude * p_vetor

                    # Desenha a flecha no gráfico
                    ax.quiver(x - u, z - w, y - v, u, w, v,
                              color='magenta',
                              arrow_length_ratio=0.2, linewidth=1.5)

                    # Adiciona o valor ao gráfico
                    ax.text(x - u, z - w, y - v,
                            f'{carga:.2f} kN', color='magenta', fontsize=16, fontweight='bold', ha='center', va='bottom')


def plot_momentos_concentrados(ax, estrutura, M, raio=0.25):
    """
    Plota os momentos concentrados nos nós da estrutura.
    """
    def calcular_arco_e_seta(j, direction, x, y, z, raio):
        """
        Calcula as coordenadas do arco e da seta para representar o momento.
        """
        angles = np.linspace(0, 7 * np.pi / 4, 100) if direction == 'ccw' \
            else np.linspace(np.pi, -3 * np.pi / 4, 100)

        if j == 0:  # Momento no eixo X
            arc_x = np.full_like(angles, x)
            arc_y = y + raio * np.cos(angles)
            arc_z = z + raio * np.sin(angles)
            dx, dy, dz = 0, -raio * np.sin(angles[-1]), raio * np.cos(angles[-1])
        elif j == 1:  # Momento no eixo Y
            arc_x = x + raio * np.cos(angles)
            arc_y = np.full_like(angles, y)
            arc_z = z + raio * np.sin(angles)
            dx, dy, dz = -raio * np.sin(angles[-1]), 0, raio * np.cos(angles[-1])
        else:  # Momento no eixo Z
            arc_x = x + raio * np.cos(angles)
            arc_y = y + raio * np.sin(angles)
            arc_z = np.full_like(angles, z)
            dx, dy, dz = -raio * np.sin(angles[-1]), raio * np.cos(angles[-1]), 0

        return arc_x, arc_y, arc_z, dx, dy, dz

    for i, (x, y, z) in enumerate(estrutura.coord):
        if np.any(M[i] != 0):  # Verifica se há momentos não nulos
            for j, momento in enumerate(M[i]):
                if momento != 0:
                    direction = 'cw' if momento > 0 else 'ccw'
                    color = 'indigo'

                    # Calcula o arco e a seta
                    arc_x, arc_y, arc_z, dx, dy, dz = calcular_arco_e_seta(j, direction, x, y, z, raio)

                    # Plota o arco 3D
                    ax.plot(arc_x, arc_z, arc_y, color=color, linewidth=2)

                    # Adiciona a seta no final do arco
                    ax.quiver(arc_x[-1], arc_z[-1], arc_y[-1],
                              dx * 0.5, dz * 0.5, dy * 0.5,
                              color=color, arrow_length_ratio=1)

                    # Adiciona o valor ao gráfico
                    x_plot, y_plot, z_plot = x, y, z
                    if j == 0:
                        x_plot += raio
                    elif j == 1:
                        y_plot += raio
                    else:
                        z_plot += raio

                    ax.text(x_plot, y_plot, z_plot,
                            f'{momento:.2f} kNm', color=color, fontsize=14, ha='center', va='bottom')


def malha_indeformada(coord, estrutura, ref_vector):
    """
    Função para criar a malha da estrutura indeformada (PyVista).
    """
    # Inverter os pontos de y e z
    pontos = coord[:, :, [0, 2, 1]]

    # Paralelizar a criação das malhas
    malha_estrutura = Parallel(n_jobs=-1)(
        delayed(criar_malhas_em_paralelo)(i, pontos, estrutura, ref_vector) 
        for i in tqdm(range(coord.shape[0]), desc="Criando malhas da estrutura indeformada"))

    # Desempacotar os resultados em listas separadas
    tubos, malhas = zip(*malha_estrutura)

    # Converter listas para MultiBlock
    tubos_ind, secao_ind = map(pv.MultiBlock, (tubos, malhas))

    return tubos_ind, secao_ind


def malha_deformada(coords_deformadas, elementos, estrutura, num_modos, ref_vector):
    """
    Função para criar a malha da estrutura deformada (PyVista).

    Args
    ----------
    coords_deformadas : np.ndarray
        Coordenadas deformadas da estrutura
    estrutura : np.ndarray
        Vetor com as informações da estrutura (número de elementos, nodes, etc.)
    ref_vector : np.ndarray
        Vetor de referência para a estrutura

    Returns
    -------
    tubos_def : dict
        Dicionário com os tubos da estrutura deformada, separados por tipo de análise
    secao_def : dict
        Dicionário com as malhas da estrutura deformada, separadas por tipo de análise
    """
    # Inicializar dicionários para armazenar os resultados
    tubos_def = {
        'linear': [None] * elementos,
        'não-linear': [None] * elementos,
        'flambagem': [[None] * elementos for _ in range(num_modos)]
    }
    secao_def = {
        'linear': [None] * elementos,
        'não-linear': [None] * elementos,
        'flambagem': [[None] * elementos for _ in range(num_modos)]
    }

    # Paralelizar a criação das malhas
    with Parallel(n_jobs=-1) as parallel:
        results = parallel(delayed(criar_malhas_em_paralelo)(i, coords_deformadas, estrutura, ref_vector, deformada=True)
                          for i in tqdm(range(elementos), desc="Criando malhas da estrutura"))

    # Separar os resultados por tipo de análise
    for i, grid in enumerate(results):
        if grid is not None:
            tubo, malha = grid
            tubos_def['linear'][i] = tubo.get('linear')
            tubos_def['não-linear'][i] = tubo.get('não-linear')
            secao_def['linear'][i] = malha.get('linear')
            secao_def['não-linear'][i] = malha.get('não-linear')

            for modo in range(num_modos):
                if modo < len(tubo.get('flambagem', [])):
                    tubos_def['flambagem'][modo][i] = tubo['flambagem'][modo]
                    secao_def['flambagem'][modo][i] = malha['flambagem'][modo]

    return tubos_def, secao_def


def plot_estrutura_matplotlib(ax, estrutura, coord, conec, q, P, M,
                              magnitude, transparencia=1.0, plotar_nos=True, legenda=False):
    # Inverter os eixos y <-> z
    coordenadas = coord[conec][:, :, [0, 2, 1]]

    # Criar uma lista de segmentos de linha
    segmentos = np.array([coordenadas[i, [0, 1]] for i in range(coordenadas.shape[0])])

    # Plotar todos os segmentos de uma vez
    linhas = Line3DCollection(segmentos, colors='k', linewidths=3, alpha=transparencia)
    ax.add_collection3d(linhas)

    # Plotar nós, se solicitado
    if plotar_nos:
        ax.scatter(coordenadas[:, :, 0], coordenadas[:, :, 1], coordenadas[:, :, 2],
                   color='darkblue', s=64, alpha=transparencia)

    # Adicionar legenda, se solicitado
    if legenda:
        ax.scatter([], [], [], color='darkblue', s=64, label='Nós')
        ax.plot([], [], [], color='k', linewidth=3, label='Elementos')

        # Adicionar texto para cada nó
        nos_unicos = np.unique(coordenadas.reshape(-1, 3), axis=0)
        for idx, (x, y, z) in enumerate(nos_unicos):
            ax.text(x, y + 0.1, z, f"{idx+1}",
                    fontsize=16, fontweight="bold", color="darkblue", ha="center", va="bottom")

        ax.legend(loc='upper right', fontsize=12)

    # Plotar cargas distribuídas
    plot_cargas_distribuidas(ax, coord, conec, q, magnitude)

    # Plotar cargas concentradas
    plot_cargas_concentradas(ax, estrutura, P, magnitude)

    # Plotar momentos concentradas
    plot_momentos_concentrados(ax, estrutura, M, magnitude)

    # Representar os apoios
    plot_apoio(ax, None, estrutura, 'matplotlib')

    # Definir limites dos eixos
    limites = np.array([coordenadas.min(axis=(0, 1)), coordenadas.max(axis=(0, 1))])
    margem = 0.25 * np.ptp(limites, axis=0).max()
    limites_com_margem = np.array([limites[0] - margem, limites[1] + margem]).T
    x_lim, z_lim, y_lim = limites_com_margem

    # Ajustar os limites dos eixos
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    # Esconder os ticks dos eixos
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Esconder as linhas do grid
    ax.grid(False)

    # Plotar a estrutura
    plt.show()


def plot_estrutura_pyvista(grid_tubos, grid_secao, plotter, estrutura, transparencia=0.5, plotar_secao=True):
    """
    Função para plotar a estrutura indeformada (PyVista)
    """

    # Adicionar as malhas à estrutura
    plotter.add_mesh(grid_tubos, color='gray', opacity=transparencia)

    if plotar_secao:
        plotter.add_mesh(grid_secao, color='lightblue', show_edges=False)

    # Plotar os apoios
    plot_apoio(None, plotter, estrutura, 'pyvista')

    # Adicionar título ao gráfico
    plotter.add_text(f"{estrutura.nome}", position='upper_right', font_size=14)

    # Plotar a estrutura
    plotter.show()
