# Bibliotecas
import numpy as np
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.signal import argrelextrema


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

    # Adicionar eixos ao gráfico
    plotter.show_axes()


def coordenadas_deformadas(coords, dl, dnl, d_flamb, MT, tipo, escala):
    """
    Função para obter as coordenadas deformadas ordenadas.

    Parâmetros:
        tipo (str): Tipo de análise ('linear', 'não-linear', 'flambagem').
        escala (int): Escala a ser aplicada nos deslocamentos.

    Retorna:
        np.ndarray: Matriz das coordenadas deformadas (elementos, 2, 3).
    """
    # Parâmetros iniciais
    elementos = dl.shape[0]
    num_modos = d_flamb.shape[0]

    # Verificar o tipo de deformação
    if tipo == 'linear':
        # Deformação local
        dloc = np.stack((dl[:, :3], dl[:, 6:9]), axis=1).squeeze()

        # Deformação global
        dglob = np.einsum('eji,enj->eni', MT, dloc, optimize=True)

        # Coordenadas deformadas
        coord = coords + escala * dglob

        # Inverter os eixos y <-> z
        coord[:, :, [1, 2]] = coord[:, :, [2, 1]]

    elif tipo == 'não-linear':
        # Deformação local
        dloc = np.stack((dnl[:, :3], dnl[:, 6:9]), axis=1).squeeze()

        # Deformação global
        dglob = np.einsum('eji,enj->eni', MT, dloc, optimize=True)

        # Coordenadas deformadas
        coord = coords + escala * dglob

        # Inverter os eixos y <-> z
        coord[:, :, [1, 2]] = coord[:, :, [2, 1]]

    elif tipo == 'flambagem':
        # Vetorização para flambagem
        coord = np.zeros((elementos, num_modos, 2, 3))

        for idx in range(num_modos):
            # Deformação local
            dloc = np.stack((d_flamb[idx][:, :3], d_flamb[idx][:, 6:9]), axis=1).squeeze()

            # Deformação global
            dglob = np.einsum('eji,enj->eni', MT, dloc, optimize=True)

            # Coordenadas deformadas
            coord[:, idx] = coords + escala * dglob

        # Inverter os eixos y <-> z
        coord[:, :, :, [1, 2]] = coord[:, :, :, [2, 1]]

    else:
        raise ValueError(f"Tipo de deformação '{tipo}' não suportado.")

    return coord


def calcular_deslocamentos(desl_linear, desl_nao_linear, desl_flambagem, MT, tipo, modo=None):
    """
    Calcula os deslocamentos para todos os elementos de forma vetorizada.
    
    Args:
        tipo (str): Tipo de análise ('linear', 'não-linear', 'flambagem').
        modo (int): Modo de flambagem (apenas para tipo='flambagem').
    
    Returns:
        np.ndarray: Matriz de deslocamentos com shape (elementos, n, 3).
    """
    if tipo == 'linear':
        pontos = np.stack((desl_linear['u'], desl_linear['v'], desl_linear['w']), axis=-1)

    elif tipo == 'não-linear':
        pontos = np.stack((desl_nao_linear['u'], desl_nao_linear['v'], desl_nao_linear['w']), axis=-1)

    elif tipo == 'flambagem':
        pontos = np.stack((desl_flambagem['u'][modo], desl_flambagem['v'][modo], desl_flambagem['w'][modo]), axis=-1)

    else:
        raise ValueError(f"Tipo de análise desconhecido: {tipo}")

    # Transformar para o eixo global
    deslocamentos = np.einsum('eji,enj->eni', MT, pontos, optimize=True)

    return deslocamentos


def adicionar_escalars(i, malha, valores_iniciais, valores_finais):
    if malha is not None:
        scalars = np.linspace(valores_iniciais[i], valores_finais[i], malha.n_points)
        malha["scalars"] = scalars
    return malha


def deslocamentos_plotagem(elementos, num_modos, dl, dnl, d_flamb):
    """
    Calcular os deslocamentos lineares, não-lineares e de flambagem-linear para a plotagem.

    Args:
        dl (np.ndarray): Deslocamentos lineares.
        dnl (np.ndarray): Deslocamentos não-lineares.
        d_flamb (np.ndarray): Deslocamentos de flambagem.

    Returns:
        deslocamentos_lineares (dict): Dicionário com os deslocamentos lineares.
        deslocamentos_nao_lineares (dict): Dicionário com os deslocamentos não-lineares.
        deslocamentos_flambagem (dict): Dicionário com os deslocamentos de flambagem.
    """

    # Inicializar dicionários para armazenar os resultados
    deslocamentos_lineares = {key: np.zeros((elementos, 2)) for key in ['u', 'v', 'w', 'θx', 'θy', 'θz']}
    deslocamentos_nao_lineares = {key: np.zeros((elementos, 2)) for key in ['u', 'v', 'w', 'θx', 'θy', 'θz']}
    deslocamentos_flambagem = {key: np.zeros((num_modos, elementos, 2)) for key in ['u', 'v', 'w', 'θx', 'θy', 'θz']}

    # Obtenção dos deslocamentos nodais lineares
    deslocamentos_lineares['u'] = dl[:, [0, 6], 0]
    deslocamentos_lineares['v'] = dl[:, [1, 7], 0]
    deslocamentos_lineares['w'] = dl[:, [2, 8], 0]
    deslocamentos_lineares['θx'] = dl[:, [3, 9], 0]
    deslocamentos_lineares['θy'] = dl[:, [4, 10], 0]
    deslocamentos_lineares['θz'] = dl[:, [5, 11], 0]

    # Obtenção dos deslocamentos nodais não lineares
    deslocamentos_nao_lineares['u'] = dnl[:, [0, 6], 0]
    deslocamentos_nao_lineares['v'] = dnl[:, [1, 7], 0]
    deslocamentos_nao_lineares['w'] = dnl[:, [2, 8], 0]
    deslocamentos_nao_lineares['θx'] = dnl[:, [3, 9], 0]
    deslocamentos_nao_lineares['θy'] = dnl[:, [4, 10], 0]
    deslocamentos_nao_lineares['θz'] = dnl[:, [5, 11], 0]

    # Obtenção dos deslocamentos nodais de flambagem
    deslocamentos_flambagem['u'] = d_flamb[:, :, [0, 6], 0]
    deslocamentos_flambagem['v'] = d_flamb[:, :, [1, 7], 0]
    deslocamentos_flambagem['w'] = d_flamb[:, :, [2, 8], 0]
    deslocamentos_flambagem['θx'] = d_flamb[:, :, [3, 9], 0]
    deslocamentos_flambagem['θy'] = d_flamb[:, :, [4, 10], 0]
    deslocamentos_flambagem['θz'] = d_flamb[:, :, [5, 11], 0]

    return deslocamentos_lineares, deslocamentos_nao_lineares, deslocamentos_flambagem


def plotar_deslocamentos(tubos_ind, malha_ind, tubos_def, malha_def, estrutura, dl, dnl, d_flamb,
                         MT, autovalores, analise, eixo, modo, coords_deformadas, widget):
    """
    Função para plotar os deslocamentos da estrutura.

    Parâmetros:
        tubos_ind (list): Lista de tubos da estrutura indeformada.
        malha_ind (list): Lista de malhas da estrutura indeformada.
        tubos_def (list): Lista de tubos da estrutura deformada.
        malha_def (list): Lista de malhas da estrutura deformada.
        estrutura (Estrutura): Instância da classe Estrutura.
        dl (np.ndarray): Deslocamentos lineares.
        dnl (np.ndarray): Deslocamentos não-lineares.
        d_flamb (np.ndarray): Deslocamentos de flambagem.
        MT (np.ndarray): Matriz de transformação.
        autovalores (list): Lista com os valores de autovalores.
        analise (str): Tipo de análise ('linear', 'não-linear' ou 'flambagem').
        eixo (str): Eixo de deslocamento ('UX', 'UY', 'UZ', 'U').
        modo (int): Modo de flambagem.
        coords_deformadas (np.ndarray): Coordenadas deformadas da estrutura.
        widget (pyvista.Plotter): Widget para plotar a estrutura.
    """

    # Resetar o widget
    widget.clear_actors()

    # Definir cor de fundo
    widget.set_background("darkgray", top='white')

    # Plotar a estrutura indeformada
    plot_estrutura_pyvista(tubos_ind, malha_ind, widget, estrutura, transparencia=0.50, plotar_secao=False)

    # Calcular os deslocamentos para a plotagem
    desl_linear, desl_nao_linear, desl_flambagem = deslocamentos_plotagem(dl.shape[0], len(autovalores), dl, dnl, d_flamb)

    # Calcular deformação para todos os elementos
    deslocamentos = calcular_deslocamentos(desl_linear, desl_nao_linear, desl_flambagem, MT, analise, modo)

    # Selecionar os valores dos deslocamentos com base no eixo
    eixo_map = {'UX': (0, 'UX (m)'), 'UY': (1, 'UY (m)'), 'UZ': (2, 'UZ (m)'), 'U': (slice(None), 'U (m)')}
    try:
        eixo_idx, title = eixo_map[eixo]
        valores = np.linalg.norm(deslocamentos, axis=2) if eixo == 'U' else deslocamentos[:, :, eixo_idx]
    except KeyError:
        raise ValueError(f"Eixo '{eixo}' não suportado.")

    # Valores dos deslocamentos para o cmap
    valores_iniciais, valores_finais = valores[:, 0], valores[:, -1]
    vmin, vmax = valores.min(), valores.max()

    # Desempacotar as malhas em listas separadas
    tubos = tubos_def[analise] if analise in ['linear', 'não-linear'] else tubos_def[analise][modo]
    secoes = malha_def[analise] if analise in ['linear', 'não-linear'] else malha_def[analise][modo]

    # Transformar as malhas para o formato MultiBlock
    grid_tubos = pv.MultiBlock(tubos)
    grid_secao = pv.MultiBlock(
        Parallel(n_jobs=-1)(
            delayed(adicionar_escalars)(i, malha, valores_iniciais, valores_finais)
            for i, malha in enumerate(secoes)
        )
    )

    # Configurar argumentos do mapa de cores
    scalar_bar_args = {
        'title': title,
        'title_font_size': 24,
        'label_font_size': 20,
        'n_labels': 10,
        'vertical': True,
        'fmt': '%.3e'
    }

    # Adicionar as malhas ao plotter
    widget.add_mesh(grid_tubos, color='sienna', opacity=1.0)
    widget.add_mesh(grid_secao, scalars='scalars', cmap="jet", clim=[vmin, vmax], scalar_bar_args=scalar_bar_args)

    # Armazenar a posição do deslocamento máximo
    pos_max_deslocamento = None

    # Encontrar a posição do deslocamento máximo
    idx_max_global = np.unravel_index(np.abs(valores).argmax(), valores.shape)
    if analise in ['linear', 'não-linear']:
        pos_max_deslocamento = coords_deformadas[analise][idx_max_global[0], idx_max_global[1]]
    else:
        pos_max_deslocamento = coords_deformadas[analise][:, modo][idx_max_global[0], idx_max_global[1]]

    # Adicionar marcador e rótulo para o deslocamento máximo
    if pos_max_deslocamento is not None:
        esfera = pv.Sphere(radius=0.1, center=pos_max_deslocamento)
        widget.add_mesh(esfera, color='black', opacity=1.0)
        widget.add_point_labels(
            pos_max_deslocamento,
            [f'Max: {valores[idx_max_global]:.3f}'],
            font_size=28,
            text_color='red',
            shape_color='white',
            shape_opacity=0.8,
            margin=5,
            always_visible=True,
        )

    # Definir e adicionar o título do gráfico
    title_map = {
        'linear': 'Deslocamentos nodais (elástico-linear)',
        'não-linear': 'Deslocamentos nodais (não-linearidade geométrica)',
        'flambagem': f'Flambagem linear, modo {modo + 1} = {autovalores[modo]:.3f}'
    }
    widget.add_text(title_map[analise], position='upper_left', font_size=14)


def plotar_esforcos(tubos_ind, malha_ind, tubos_def, malha_def, elementos, estrutura, xp, coords, 
                    esforcos_lineares, esforcos_nao_lineares, MT, pontos_int, biblioteca, analise, esforco, escala, widget):
    # Determinar os valores dos esforços
    if analise == 'linear':
        esforcos = {
            'Fx': (-esforcos_lineares['Fx'], 'Fx (kN)'),
            'Fy': (esforcos_lineares['Fy'], 'Fy (kN)'),
            'Fz': (esforcos_lineares['Fz'], 'Fz (kN)'),
            'Mx': (esforcos_lineares['Mx'], 'Mx (kN.m)'),
            'My': (esforcos_lineares['My'], 'My (kN.m)'),
            'Mz': (esforcos_lineares['Mz'], 'Mz (kN.m)'),
        }
    else:
        esforcos = {
            'Fx': (-esforcos_nao_lineares['Fx'], 'Fx (kN)'),
            'Fy': (esforcos_nao_lineares['Fy'], 'Fy (kN)'),
            'Fz': (esforcos_nao_lineares['Fz'], 'Fz (kN)'),
            'Mx': (esforcos_nao_lineares['Mx'], 'Mx (kN.m)'),
            'My': (esforcos_nao_lineares['My'], 'My (kN.m)'),
            'Mz': (esforcos_nao_lineares['Mz'], 'Mz (kN.m)'),
        }
    
    if biblioteca == 'matplotlib':
        ax = widget.figure.gca()
        ax.clear()

        # Dicionário de esforços por tipo
        dados = {
            'Fx': {'color': 'b', 'direction': 1, 'eixo': 1},
            'Fy': {'color': 'g', 'direction': 1, 'eixo': 1},
            'Fz': {'color': 'r', 'direction': 1, 'eixo': 2},
            'Mx': {'color': 'peru', 'direction': -1, 'eixo': 1},
            'My': {'color': 'purple', 'direction': -1, 'eixo': 2},
            'Mz': {'color': 'slateblue', 'direction': -1, 'eixo': 1}
        }

        # Obtém os valores do esforço selecionado
        valores, title = esforcos[esforco]

        # Magnitude unitária dos esforços
        magnitude = 1 / np.max(np.abs(valores), initial=1)

        # Escalar os esforços de acordo com dados do usuário
        f_escalado = (magnitude * valores) * escala

        # Define o rótulo para os esforços
        def label(i, idx):
            return f'{valores[i, idx]:.3f}'

        # Inverter a direção do gráfico, caso necessário
        direction = dados[esforco]['direction']

        # Obtém o eixo em que os esforços devem ser plotados
        eixo = dados[esforco]['eixo']

        # Cria a matriz de forças corretamente posicionada
        f = np.zeros((elementos, pontos_int, 3))

        # Posiciona os esforços no eixo correto
        f[:, :, eixo] = direction * f_escalado

        # Aplica a transformação para o sistema global
        f_global = np.einsum('eji,epj->epi', MT, f)

        # Coordenadas dos pontos deformados
        pontos_deformados = xp[..., [0, 2, 1]] + escala * f_global[..., [0, 2, 1]]

        # Definir a cor da plotagem
        color = dados[esforco]['color']

        # Plotagem da estrutura
        for i in range(elementos):
            # Plotar o elemento (barra)
            ax.plot([coords[i, 0, 0], coords[i, 1, 0]],
                    [coords[i, 0, 2], coords[i, 1, 2]],
                    [coords[i, 0, 1], coords[i, 1, 1]],
                    marker='o', markersize=4, color="black", linestyle='-')

            # Coordenadas dos polígonos
            polygon_coords = np.zeros((pontos_int + 1, 3))
            polygon_coords[0] = [xp[i, 0, 0], xp[i, 0, 2], xp[i, 0, 1]]
            polygon_coords[1:] = pontos_deformados[i]
            polygon_coords = np.vstack([polygon_coords, [xp[i, pontos_int - 1, 0], xp[i, pontos_int - 1, 2], xp[i, pontos_int - 1, 1]]])

            # Criar a coleção de polígonos 3D
            poly3d = Poly3DCollection([polygon_coords], closed=False, linewidths=2, edgecolors=color, facecolors=color, alpha=0.5)
            ax.add_collection3d(poly3d)

            # Adicionar valores ao gráfico
            if esforco not in ['Mx', 'My', 'Mz']:
                va1 = 'top' if float(label(i, 0)) < 0 else 'baseline'
                va2 = 'baseline' if float(label(i, pontos_int - 1)) >= 0 else 'top'
            else:
                va1 = 'baseline' if float(label(i, 0)) <= 0 else 'top'
                va2 = 'top' if float(label(i, pontos_int - 1)) > 0 else 'baseline'

            ax.text(polygon_coords[1, 0], polygon_coords[1, 1], polygon_coords[1, 2], label(i, 0), ha='center',
                    va=va1, fontsize=14)
            ax.text(polygon_coords[pontos_int, 0], polygon_coords[pontos_int, 1], polygon_coords[pontos_int, 2], label(i, pontos_int - 1), ha='center',
                    va=va2, fontsize=14)

            # Adicionar os valores máximos e mínimos
            max_idx = argrelextrema(valores[i], np.greater, order=5)[0]
            min_idx = argrelextrema(valores[i], np.less, order=5)[0]

            if max_idx.size > 0 or min_idx.size > 0:
                idx = (max_idx[0] if max_idx.size > 0 else min_idx[0])
                value = float(label(i, idx))
                if value and not idx == 0 and not idx == pontos_int - 1:
                    if esforco not in ['Mx', 'My', 'Mz']:
                        va = 'top' if value < 0 else 'baseline'
                    else:
                        va = 'baseline' if value <= 0 else 'top'
                    ax.plot([xp[i, idx, 0], polygon_coords[idx, 0]], 
                            [xp[i, idx, 2], polygon_coords[idx, 1]],
                            [xp[i, idx, 1], polygon_coords[idx, 2]], linewidth=2, color=color)
                    ax.text(polygon_coords[idx, 0], polygon_coords[idx, 1], polygon_coords[idx, 2], f'{value:.2f}', ha='center', va=va, fontsize=12)        
        
        # Adicionar um marcador invisível para incluir a legenda
        ax.plot([], [], [], color=color, label=title)

        # Adicionar o ícone do apoio ao gráfico
        plot_apoio(ax, None, estrutura, 'matplotlib')

        # Definir rótulos dos eixos
        ax.set_xlabel('Eixo X')
        ax.set_ylabel('Eixo Z')
        ax.set_zlabel('Eixo Y')

        # Adicionar legendas para os gráficos selecionados
        ax.legend(loc='lower right')

        # Ajustar a proporção dos eixos para melhorar o zoom
        ax.set_box_aspect([1, 1, 1])

        # Atualizar a exibição do gráfico
        widget.canvas.draw_idle()

    elif biblioteca == 'pyvista':
        widget.clear_actors()

        # Definir cor de fundo
        widget.set_background("darkgray", top='white')

        # Plotar a estrutura indeformada
        plot_estrutura_pyvista(tubos_ind, malha_ind, widget, estrutura, transparencia=0.5, plotar_secao=False)

        if esforco not in esforcos:
            raise ValueError(f"Tipo de esforço desconhecido: {esforco}")

        valores, title = esforcos[esforco]

        # Valores dos esforços no início e no fim de cada elemento
        valores_iniciais, valores_finais = valores[:, 0], valores[:, -1]
        vmin, vmax = valores.min(), valores.max()

        # Transformar as malhas para o formato MultiBlock
        grid_tubos = pv.MultiBlock(tubos_def[analise])
        grid_secao = pv.MultiBlock(
            Parallel(n_jobs=-1)(
                delayed(adicionar_escalars)(i, malha, valores_iniciais, valores_finais)
                for i, malha in enumerate(malha_def[analise])
            )
        )

        # Configurar argumentos do mapa de cores
        scalar_bar_args = {
            'title': title,
            'title_font_size': 24,
            'label_font_size': 20,
            'n_labels': 10,
            'vertical': True,
            'fmt': '%.3f'
        }

        # Adicionar as malhas ao plotter
        widget.add_mesh(grid_tubos, color='sienna', opacity=1.0)
        widget.add_mesh(grid_secao, scalars='scalars', cmap="jet", clim=[vmin, vmax], scalar_bar_args=scalar_bar_args)

        # Definir título do gráfico
        if analise == 'linear':
            title = 'Esforços lineares'
        elif analise == 'não-linear':
            title = 'Esforços não-lineares'

        # Adicionar título ao gráfico
        widget.add_text(title, position='upper_left', font_size=14)


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
    secao_transversal = estrutura['geometria']

    # Criar a seção transversal
    if secao_transversal == 'retangular':
        b, h = estrutura['base'], estrutura['altura']

        # Definir pontos para o perfil retangular
        pontos = np.array([[-b/2, -h/2, 0], [b/2, -h/2, 0], [b/2, h/2, 0], [-b/2, h/2, 0]])

        # Definir faces
        faces = np.array([[4, 0, 1, 2, 3]])

        # Criar a seção
        secao = pv.PolyData(pontos, faces)
    
    elif secao_transversal == 'caixa':
        b, h, t = estrutura['base'], estrutura['altura'], estrutura['espessura']

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
        secao = pv.Circle(radius=estrutura['raio'], resolution=50)

    elif secao_transversal == 'tubular':
        secao = pv.Disc(inner=estrutura['raio_int'], outer=estrutura['raio_ext'], r_res=50, c_res=50)

    elif secao_transversal == 'I':
        b, h, tf, tw = estrutura['base'], estrutura['altura'], estrutura['espessura_flange'], estrutura['espessura_alma']

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
        b, h, tf, tw = estrutura['base'], estrutura['altura'], estrutura['espessura_flange'], estrutura['espessura_alma']

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

    # Obter a estrutura do elemento 'i'
    estrutura_elemento = estrutura.secoes[i]

    # Criar a malha do elemento (deformada ou indeformada)
    if deformada:
        # Processar cada tipo de análise
        for analise in pontos:
            coords_list = pontos[analise][i] if analise == 'flambagem' else [pontos[analise][i]]
            for modo, coords in enumerate(coords_list):
                if plotar_secao:
                    malha_atual = criar_malha_elemento(coords, estrutura_elemento, ref_vector[i])
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
            malha = criar_malha_elemento(coords, estrutura_elemento, ref_vector[i])
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
