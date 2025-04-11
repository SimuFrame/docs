import numpy as np

class Estrutura():
    def __init__(self, nome, modelo, coord, conec, subdivisoes):
        self.nome = nome
        self.modelo = modelo
        self.coord = np.array(coord)
        self.conec = np.array(conec, dtype=int)
        self.coord_original = np.copy(self.coord)       # Coordenadas antes da subdivisão
        self.conec_original = np.copy(self.conec)       # Conectividade antes da subdivisão
        self.subdivisoes = max(1, subdivisoes)          # Garantir que subdivisões seja pelo menos 1
        self.secao = {}                                 # Dicionário para armazenar as propriedades geométricas das seções

        # Inicializar novos nós e conectividades
        novas_coords = list(map(tuple, self.coord))  # Converter array numpy para lista de tuplas
        novas_conec = []

        # Dicionário para mapear coordenadas para índices
        coord_to_index = {coord: i for i, coord in enumerate(novas_coords)}

        if self.subdivisoes == 1:
            novas_conec, novas_coords = self.conec, self.coord
        else:
            for no_inicial, no_final in self.conec:
                # Coordenadas dos nós do elemento
                coord_inicial = self.coord[no_inicial]
                coord_final = self.coord[no_final]

                # Determinar o número real de subdivisões
                n_divisoes = self.subdivisoes - 1
                delta = (coord_final - coord_inicial) / (n_divisoes + 1)

                # Criar novos nós e atualizar conectividades
                indices = []
                for i in range(n_divisoes + 2):
                    no_tuple = tuple(coord_inicial + i * delta)
                    index = coord_to_index.setdefault(no_tuple, len(novas_coords))  # Evita múltiplas verificações
                    if index == len(novas_coords):  # Apenas adiciona novo nó se for realmente novo
                        novas_coords.append(no_tuple)
                    indices.append(index)

                # Adicionar conectividades entre nós consecutivos
                novas_conec.extend(zip(indices[:-1], indices[1:]))

        # Converter lista de tuplas de volta para numpy array
        # novas_coords = np.array(novas_coords)

        # Atualizar as propriedades da estrutura
        self.coord = np.array(novas_coords, dtype=float)
        self.conec = np.array(novas_conec, dtype=int)
        
        # Inicializar as condições de contorno
        self.vinculacoes = [''] * len(self.coord)

        # Inicializar cargas
        self.cargas_concentradas = []       # Lista de tuplas (nó, [Px, Py, Pz])
        self.cargas_iniciais = []           # Lista de tuplas para as cargas distribuídas iniciais do elemento
        self.cargas_distribuidas = []       # Lista de tuplas (elemento, [qx1, qy1, qz1], [qx2, qy2, qz2])
        self.momentos_concentrados = []     # Lista de tuplas (nó, [Mx, My, Mz])

    def definir_apoios(self, apoios):
        """
        Define os apoios da estrutura (engaste, fixo, móvel), de acordo com os índices fornecidos.
        """
        for vinculo, indices in apoios.items():
            for idx in indices:
                self.vinculacoes[idx] = vinculo

    def CLOAD(self, no, carga):
        """
        Adiciona carga concentrada ao nó especificado, no formato [Px, Py, Pz].
        """

        self.cargas_concentradas.append((no, carga))    

    def MLOAD(self, no, carga):
        """
        Adiciona momento concentrado ao nó especificado, no formato [Mx, My, Mz].
        """

        self.momentos_concentrados.append((no, carga))

    def DLOAD(self, elemento, carga_inicio, carga_fim):
        """
        Adiciona uma carga distribuída ao elemento, considerando subdivisões.
        """
        # Salvar as cargas iniciais para fins de representação gráfica
        self.cargas_iniciais.append((elemento, carga_inicio, carga_fim))

        # Converter as cargas para arrays NumPy para operações vetorizadas
        carga_inicio = np.array(carga_inicio)
        carga_fim = np.array(carga_fim)

        if self.subdivisoes > 1:
            # Fatores de interpolação para as cargas
            fatores_inicio = np.linspace(0, 1, self.subdivisoes, endpoint=False)
            fatores_fim = np.linspace(1 / self.subdivisoes, 1, self.subdivisoes)

            # Interpolar as cargas nos subelementos
            q_inicio_sub = (1 - fatores_inicio[:, None]) * carga_inicio + fatores_inicio[:, None] * carga_fim
            q_fim_sub = (1 - fatores_fim[:, None]) * carga_inicio + fatores_fim[:, None] * carga_fim

            # Gerar subelementos e armazenar as cargas subdivididas
            cargas_subdivididas = [
                (elemento * self.subdivisoes + i, q_inicio_sub[i].tolist(), q_fim_sub[i].tolist())
                for i in range(self.subdivisoes)
            ]

            # Atualizar as cargas distribuídas com as subdivisões
            self.cargas_distribuidas.extend(cargas_subdivididas)

        else:
            # Sem subdivisões: Adicionar a carga diretamente
            self.cargas_distribuidas.append((elemento, carga_inicio.tolist(), carga_fim.tolist()))
    
    def geometria(self, tipo, **kwargs):
        """
        Define as propriedades geométricas de uma seção transversal.

        Parâmetros:
        - tipo (str): O tipo da seção (e.g., "retangular", "circular", "tubular").
        - **kwargs: Parâmetros específicos para cada tipo de seção.
        """
        # Determinar o módulo de elasticidade longitudinal e transversal do elemento
        E = kwargs.get("E")
        v = kwargs.get("v")
        G = E / (2 * (1 + v)) if E and v else None

        # Dicionário de validação de parâmetros para cada tipo de seção
        validacoes = {
            "retangular": {"base": float, "altura": float},
            "caixa": {"base": float, "altura": float, "espessura": float},
            "circular": {"raio": float},
            "tubular": {"raio_ext": float, "raio_int": float},
            "I": {"base": float, "altura": float, "espessura_flange": float, "espessura_alma": float},
            "T": {"base": float, "altura": float, "espessura_flange": float, "espessura_alma": float},
        }

        # Verificar se o tipo de seção é válido
        if tipo not in validacoes:
            raise ValueError(f"Tipo de seção '{tipo}' não reconhecido.")

        # Inicializar todas as dimensões geométricas como strings vazias
        dimensoes = {
            "base": None,
            "altura": None,
            "raio": None,
            "raio_ext": None,
            "raio_int": None,
            "espessura": None,
            "espessura_flange": None,
            "espessura_alma": None,
        }

        # Validar os parâmetros fornecidos e atualizar as dimensões relevantes
        for param, tipo_param in validacoes[tipo].items():
            valor = kwargs.get(param)
            if valor is None:
                raise ValueError(f"Para seção {tipo}, '{param}' é obrigatório.")
            if not isinstance(valor, tipo_param):
                raise ValueError(f"'{param}' deve ser do tipo {tipo_param.__name__}.")
            dimensoes[param] = valor # type: ignore

        # Criar o dicionário da seção
        self.secao = {"seção": tipo, **dimensoes, "E": E, "v": v, "G": G}