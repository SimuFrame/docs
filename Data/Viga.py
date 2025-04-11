import numpy as np
from DadosEstruturais import Estrutura

"""
Casos de estudo:    1: Viga biapoiada com uma carga concentrada (varal);
                    2: Viga biapoiada com uma carga concentrada;
                    3: Viga em balanço com uma carga concentrada;
                    4: Viga biapoiada com uma carga distribuída;

As condições de contorno são definidas conforme os termos:

    XSYMM: Simetria em x (U = θy = θz = 0)
    YSYMM: Simetria em y (V = θx = θz = 0)
    ZSYMM: Simetria em z (W = θx = θy = 0)
    XASYMM: Antissimetria em x (V = W = θx = 0)
    YASYMM: Antissimetria em y (U = W = θy = 0)
    ZASYMM: Antissimetria em z (U = V = θz = 0)
    ARTICULADO: Apoio articulado: (U = V = W = 0)
    FIXOXY: Apoio fixo em x e y: (U = V = W = θx = θy = 0)
    FIXOXZ: Apoio fixo em x e z: (U = Z = W = θx = θz = 0)
    FIXOYZ: Apoio fixo em y e z: (V = Z = W = θy = θz = 0)
    ENGASTE: Apoio engastado (U = V = W = θx = θy = θz = 0)
"""

# Casos de estudo (BiapCDFig, BiapCCFig, BalC30)
caso = 4

# Definir a discretização dos elementos
n = 64

# Adicionar cargas e definir parâmetros constitutivos e geométricos
if caso == 1:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [6, 0, 0]
    ])

    # Matriz de conectividade
    conec = np.array([
        [0, 1],
        [1, 2]
    ])

    # Definir os índices dos apoios
    condicoes_contorno = {'XSYMM': np.array([]),
                          'YSYMM': np.array([]),
                          'ZSYMM': np.array([1]),
                          'XASYMM': np.array([]),
                          'YASYMM': np.array([]),
                          'ZASYMM': np.array([]),
                          'ARTICULADO': np.array([]),
                          'FIXOXY': np.array([0, 2]),
                          'FIXOXZ': np.array([]),
                          'FIXOYZ': np.array([]),
                          'ENGASTE': np.array([])
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Varal (modelo de viga)', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    estrutura.CLOAD(1, [0, -0.05, 0])
    estrutura.geometria('circular', E=2.4e6, v=0.3, raio=0.02)

elif caso == 2:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [6, 0, 0]
    ])

    # Matriz de conectividade
    conec = np.array([
        [0, 1],
        [1, 2]
    ])

    # Definir os índices dos apoios
    condicoes_contorno = {'XSYMM': np.array([]),
                          'YSYMM': np.array([]),
                          'ZSYMM': np.array([1]),
                          'XASYMM': np.array([]),
                          'YASYMM': np.array([]),
                          'ZASYMM': np.array([]),
                          'ARTICULADO': np.array([]),
                          'FIXOXY': np.array([0, 2]),
                          'FIXOXZ': np.array([]),
                          'FIXOYZ': np.array([]),
                          'ENGASTE': np.array([])
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Viga (carga concentrada)', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    estrutura.CLOAD(1, [0, -0.5, 0])
    # estrutura.DLOAD(0, [0, -0.25, 0], [0, -0.25, 0])
    # estrutura.DLOAD(1, [0, -0.25, 0], [0, -0.25, 0])
    estrutura.geometria('retangular', E=2500, v=0.3, base=1.0, altura=0.12)

elif caso == 3:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([
        [0, 0, 0],
        [6, 0, 0]
    ])

    # Matriz de conectividade
    conec = np.array([
        [0, 1],
    ])

    # Definir os índices dos apoios
    condicoes_contorno = {'XSYMM': np.array([]),
                          'YSYMM': np.array([]),
                          'ZSYMM': np.array([]),
                          'XASYMM': np.array([]),
                          'YASYMM': np.array([]),
                          'ZASYMM': np.array([]),
                          'ARTICULADO': np.array([]),
                          'FIXOXY': np.array([]),
                          'FIXOXZ': np.array([]),
                          'FIXOYZ': np.array([]),
                          'ENGASTE': np.array([0])
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Viga em balanço', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    # Definir parâmetros constitutivos e geométricos
    estrutura.CLOAD(1, [0, -200, 0])
    estrutura.geometria('retangular', E=2.7e7, v=0.2, base=0.20, altura=0.40)

elif caso == 4:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([
        [0, 0, 0],
        [8, 0, 0]
    ])

    # Matriz de conectividade
    conec = np.array([
        [0, 1],
    ])

    # Definir os índices dos apoios
    condicoes_contorno = {'XSYMM': np.array([]),
                          'YSYMM': np.array([]),
                          'ZSYMM': np.array([]),
                          'XASYMM': np.array([]),
                          'YASYMM': np.array([]),
                          'ZASYMM': np.array([]),
                          'ARTICULADO': np.array([]),
                          'FIXOXY': np.array([0, 1]),
                          'FIXOXZ': np.array([]),
                          'FIXOYZ': np.array([]),
                          'ENGASTE': np.array([])
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Viga biapoiada', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    # Adicionar cargas
    estrutura.DLOAD(0, [0, -100, 0], [0, -100, 0])
    # estrutura.DLOAD(0, [0, 0, 100], [0, 0, 100])

    # Definir parâmetros constitutivos e geométricos
    estrutura.geometria('retangular', E=2.7e7, v=0.2, base=0.20, altura=0.40)
    # estrutura.geometria('caixa', E=2.1e8, v=0.3, base=0.20, altura=0.40, espessura=0.02)
    # estrutura.geometria('T', E=2.1e8, v=0.3, base=0.3, altura=0.4, espessura_flange=0.02, espessura_alma=0.03)
    # estrutura.geometria('circular', E=2e8, v=0.3, raio=0.1)
    # estrutura.geometria('tubular', E=2e8, v=0.3, raio_ext=0.1, raio_int=0.08)
