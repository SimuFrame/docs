# Importar bibliotecas
import numpy as np
import pyvista as pv
from joblib import Parallel, delayed
from scipy.signal import argrelextrema
from Visualização import plot_estrutura_pyvista, plot_apoio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

def plotar_deslocamentos(tubos_ind, malha_ind, tubos_def, malha_def, estrutura, desl_linear, desl_nao_linear, desl_flambagem,
                         MT, autovalores, analise, eixo, modo, coords_deformadas, widget):
    widget.clear_actors()

    # Definir cor de fundo
    widget.set_background("darkgray", top='white')

    # Plotar a estrutura indeformada
    plot_estrutura_pyvista(tubos_ind, malha_ind, widget, estrutura, transparencia=0.50, plotar_secao=False)

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
    if biblioteca == 'matplotlib':
        ax = widget.figure.gca()
        ax.clear()

        # Dicionário de esforços por tipo
        esforcos = {
            'Fx': {'linear': esforcos_lineares['Fx'], 'não-linear': esforcos_nao_lineares['Fx'],
                    'title': 'Fx (kN)', 'color': 'b', 'inverter': True, 'direction': 1, 'eixo': 1},
            'Fy': {'linear': esforcos_lineares['Fy'], 'não-linear': esforcos_nao_lineares['Fy'],
                   'title': 'Fy (kN)', 'color': 'g', 'inverter': False, 'direction': 1, 'eixo': 1},
            'Fz': {'linear': esforcos_lineares['Fz'], 'não-linear': esforcos_nao_lineares['Fz'], 
                   'title': 'Fz (kN)', 'color': 'r', 'inverter': False, 'direction': 1, 'eixo': 2},
            'Mx': {'linear': esforcos_lineares['Mx'], 'não-linear': esforcos_nao_lineares['Mx'],
                   'title': 'Mx (kN.m)', 'color': 'peru', 'inverter': False, 'direction': -1, 'eixo': 1},
            'My': {'linear': esforcos_lineares['My'], 'não-linear': esforcos_nao_lineares['My'],
                   'title': 'My (kN.m)', 'color': 'purple', 'inverter': False, 'direction': -1, 'eixo': 2},
            'Mz': {'linear': esforcos_lineares['Mz'], 'não-linear': esforcos_nao_lineares['Mz'],
                   'title': 'Mz (kN.m)', 'color': 'slateblue', 'inverter': False, 'direction': -1, 'eixo': 1}
        }

        # Obtém os dados do esforço selecionado
        dados = esforcos[esforco]

        # Escolhe entre linear ou não-linear
        f = dados[analise]

        # Magnitude unitária dos esforços
        magnitude = 1 / np.max(np.abs(f), initial=1)

        # Escalar os esforços de acordo com dados do usuário
        f_escalado = (magnitude * f) * escala

        # Ajusta a inversão de sinal, se necessário
        if dados['inverter']:
            f_escalado *= -1

        # Define o rótulo para os esforços
        def label(i, idx):
            if dados['inverter']:
                return f'{-dados[analise][i, idx]:.3f}'
            else:
                return f'{dados[analise][i, idx]:.3f}'

        # Inverter a direção do gráfico, caso necessário
        direction = dados['direction']

        # Obtém o eixo em que os esforços devem ser plotados
        eixo = dados['eixo']

        # Cria a matriz de forças corretamente posicionada
        f = np.zeros((elementos, pontos_int, 3))

        # Posiciona os esforços no eixo correto
        f[:, :, eixo] = direction * f_escalado

        # Aplica a transformação para o sistema global
        f_transformed = np.einsum('eij,epj->epi', MT.transpose(0, 2, 1), f)

        # Coordenadas dos pontos deformados
        pontos_deformados = xp[..., [0, 2, 1]] + escala * f_transformed[..., [0, 2, 1]]

        # Definir a cor da plotagem
        color = dados['color']

        # Definir o título do gráfico
        title = dados['title']

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
            valores = f_transformed[i, :, 0] if esforco in ['Fx'] else f_transformed[i, :, 1] if esforco in ['Fy'] else f_transformed[i, :, 2]
            max_idx = argrelextrema(valores, np.greater, order=5)[0]
            min_idx = argrelextrema(valores, np.less, order=5)[0]

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
