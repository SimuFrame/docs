import numpy as np
from DadosEstruturais import Estrutura

"""
Casos de estudo:    Pórtico 1: Viga biapoiada com uma carga distribuída sobre toda a viga, utilizando parâmetros constitutivos de FIGUEIRAS;
                    Pórtico 2: Viga biapoiada com uma carga concentrada no centro, utilizando parâmetros constitutivos de FIGUEIRAS;
                    Pórtico 3: Viga em balanço com uma carga concentrada na extremidade livre, utilizando parâmetros do concreto C30;
                    Pórtico 4: Viga biapoiada com uma carga concentrada no centro, utilizando parâmetros do concreto C30;

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

# Casos de estudo: selecione o número correspondente
caso = 2

# Definir a discretização dos elementos
n = 64

# Adicionar cargas e definir parâmetros constitutivos e geométricos
if caso == 1:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([[0, 0, 0],
                      [4, 0, 0],
                      [0, 0, 4],
                      [4, 0, 4],
                      [0, 4, 0],
                      [4, 4, 0],
                      [0, 4, 4],
                      [4, 4, 4]])

    # Matriz de conectividade
    conec = np.array([[0, 4],
                      [1, 5],
                      [2, 6],
                      [3, 7],
                      [4, 5],
                      [6, 7],
                      [4, 6],
                      [5, 7]])

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
                          'ENGASTE': np.array([0, 1, 2, 3])
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Pórtico 1', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    # Adicionar cargas
    # estrutura.CLOAD(4, [0, -100, 0])
    # estrutura.CLOAD(5, [0, -100, 0])
    # estrutura.CLOAD(6, [0, -100, 0])
    # estrutura.CLOAD(7, [0, -100, 0])
    estrutura.DLOAD(4, [0, -100, 0], [0, -100, 0])
    # estrutura.DLOAD(5, [0, -100, 0], [0, -100, 0])
    # estrutura.DLOAD(6, [0, -100, 0], [0, -100, 0])
    # estrutura.DLOAD(7, [0, -100, 0], [0, -100, 0])

    # Definir parâmetros constitutivos e geométricos
    estrutura.geometria('retangular', E=2.7e7, v=0.2, base=0.2, altura=0.2)
    # estrutura.geometria('circular', E=2.7e7, v=0.2, raio=0.1)

elif caso == 2:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([
                    [0, 0, 0],
                    [5, 0, 0],
                    [10, 0, 0],

                    [0, 0, 5],
                    [5, 0, 5],
                    [10, 0, 5],

                    [0, 0, 10],
                    [5, 0, 10],
                    [10, 0, 10],

                    [0, 3, 0],
                    [5, 3, 0],
                    [10, 3, 0],

                    [0, 3, 5],
                    [5, 3, 5],
                    [10, 3, 5],

                    [0, 3, 10],
                    [5, 3, 10],
                    [10, 3, 10],

                    [0, 6, 0],
                    [5, 6, 0],
                    [10, 6, 0],

                    [0, 6, 5],
                    [5, 6, 5],
                    [10, 6, 5],

                    [0, 6, 10],
                    [5, 6, 10],
                    [10, 6, 10]])

    # Matriz de conectividade
    conec = np.array([
                    # Pilares (primeiro pavimento)
                    [0, 9],
                    [1, 10],
                    [2, 11],
                    [3, 12],
                    [4, 13],
                    [5, 14],
                    [6, 15],
                    [7, 16],
                    [8, 17],

                    # Pilares (segundo pavimento)
                    [9, 18],
                    [10, 19],
                    [11, 20],
                    [12, 21],
                    [13, 22],
                    [14, 23],
                    [15, 24],
                    [16, 25],
                    [17, 26],

                    # Vigas (eixo x)
                    [9, 10],
                    [10, 11],
                    [12, 13],
                    [13, 14],
                    [15, 16],
                    [16, 17],
                    [18, 19],
                    [19, 20],
                    [21, 22],
                    [22, 23],
                    [24, 25],
                    [25, 26],

                    # Vigas (eixo z)
                    [9, 12],
                    [12, 15],
                    [10, 13],
                    [13, 16],
                    [11, 14],
                    [14, 17],
                    [18, 21],
                    [21, 24],
                    [19, 22],
                    [22, 25],
                    [20, 23],
                    [23, 26]])

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
                          'ENGASTE': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Pórtico 2', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    # Adicionar cargas
    for i in range(18, 42):
        estrutura.DLOAD(i, [0, -20, 0], [0, -20, 0])

    for i in (0, 1, 2, 9, 10, 11):
        estrutura.DLOAD(i, [0, 0, 10], [0, 0, 10])

    for i in (18, 19, 24, 25):
        estrutura.DLOAD(i, [0, -20, 10], [0, -20, 10])

    # Definir parâmetros constitutivos e geométricos
    estrutura.geometria('retangular', E=2.7e7, v=0.2, base=0.2, altura=0.4)

elif caso == 3:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([[0, 0, 0],
                      [0, 3, 0],
                      [4, 3, 0],
                      [4, 0, 0]])

    # Matriz de conectividade
    conec = np.array([[0, 1],
                      [1, 2],
                      [2, 3]])

    # Definir os índices dos apoios
    condicoes_contorno = {'XSYMM': np.array([]),
                          'YSYMM': np.array([]),
                          'ZSYMM': np.array([1, 2]),
                          'XASYMM': np.array([]),
                          'YASYMM': np.array([]),
                          'ZASYMM': np.array([]),
                          'ARTICULADO': np.array([]),
                          'FIXOXY': np.array([]),
                          'FIXOXZ': np.array([]),
                          'FIXOYZ': np.array([]),
                          'ENGASTE': np.array([0, 3])
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Pórtico 3', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    # Adicionar cargas
    estrutura.DLOAD(1, [0, -100, 0], [0, -100, 0])

    # Definir parâmetros constitutivos e geométricos
    estrutura.geometria('retangular', E=2.7e7, v=0.2, base=0.2, altura=0.2)

elif caso == 4:
    # Matriz de coordenadas dos pontos (x, y, z)
    coord = np.array([
            # Base do pórtico (nível do chão)
            [0, 0, 0],
            [5, 0, 0],
            [10, 0, 0],
            [15, 0, 0],

            [0, 0, 4],
            [5, 0, 4],
            [10, 0, 4],
            [15, 0, 4],

            [0, 0, 8],
            [5, 0, 8],
            [10, 0, 8],
            [15, 0, 8],

            # Colunas verticais (elevação)
            [0, 6, 0],
            [5, 7, 0],
            [7.5, 7.5, 0],
            [10, 7, 0],
            [15, 6, 0],

            [0, 6, 4],
            [5, 7, 4],
            [7.5, 7.5, 4],
            [10, 7, 4],
            [15, 6, 4],

            [0, 6, 8],
            [5, 7, 8],
            [7.5, 7.5, 8],
            [10, 7, 8],
            [15, 6, 8]])

    # Matriz de conectividade
    conec = np.array([
            # Colunas verticais
            [0, 12],
            [1, 13],
            [2, 15],
            [3, 16],
            [4, 17],
            [5, 18],
            [6, 20],
            [7, 21],
            [8, 22],
            [9, 23],
            [10, 25],
            [11, 26],

            # Traves horizontais superiores
            [12, 17],
            [17, 22],
            [13, 18],
            [18, 23],
            [14, 19],
            [19, 24],
            [15, 20],
            [20, 25],
            [16, 21],
            [21, 26],

            # Traves diagonais superiores
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [22, 23],
            [23, 24],
            [24, 25],
            [25, 26]])

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
                          'ENGASTE': np.array(range(12))
                          }

    # Definir o modelo estrutural (viga ou treliça)
    modelo = "viga"

    # Criar os dados do elemento estrutural
    estrutura = Estrutura('Pórtico 4', modelo, coord, conec, n)
    estrutura.definir_apoios(condicoes_contorno)

    # Adicionar cargas
    for i in range(12, 34):
        estrutura.DLOAD(i, [0, -10, 0], [0, -10, 0])

    # Definir parâmetros constitutivos e geométricos
    estrutura.geometria('retangular', E=2.7e7, v=0.2, base=0.2, altura=0.2)