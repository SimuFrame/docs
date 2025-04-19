import numpy as np
from DadosEstruturais import Estrutura


"""
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

# Matriz de coordenadas dos pontos (x, y, z)
coord = np.array([[0, 0, 0],
                  [0, 6, 0]])

# Matriz de conectividade
conec = np.array([[0, 1]])

# Definir os índices dos apoios
condicoes_contorno = {'XSYMM': [],
                      'YSYMM': [],
                      'ZSYMM': [],
                      'XASYMM': [],
                      'YASYMM': [],
                      'ZASYMM': [],
                      'ARTICULADO': [],
                      'FIXOXY': [],
                      'FIXOXZ': [],
                      'FIXOYZ': [],
                      'ENGASTE': [0]
                      }

# Definir o modelo estrutural (viga ou treliça)
modelo = "viga"

# Definir o número de subdivisões
n = 64

# Criar os dados do elemento estrutural
estrutura = Estrutura('Pilar', modelo, coord, conec, n)
estrutura.definir_apoios(condicoes_contorno)

# Adicionar cargas
estrutura.CLOAD({
    1: [0, -1, 0]
})

# Definir parâmetros constitutivos e geométricos
estrutura.geometria({
    range(len(conec)): {"geometria": "I", "E": 2.1e8, "v": 0.3, "base": 0.3, "altura": 0.4, "espessura_flange": 0.02, "espessura_alma": 0.03},
})
