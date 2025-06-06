import numpy as np

class Estrutura():
    def __init__(self, analise, nome, modelo, coord, conec, subdivisoes):
        self.analise = analise
        self.nome = nome
        self.modelo = modelo
        self.coord = np.array(coord)
        self.conec = np.array(conec, dtype=int)
        self.coord_original = np.copy(self.coord)       # Coordenadas antes da subdivisão
        self.conec_original = np.copy(self.conec)       # Conectividade antes da subdivisão
        self.subdivisoes = max(1, subdivisoes)          # Garantir que subdivisões seja pelo menos 1

        # Inicializar novos nós e conectividades
        novas_coords = list(map(tuple, self.coord))  # Converter array numpy para lista de tuplas
        novas_conec = []

        # Dicionário para mapear coordenadas para índices
        coord_to_index = {coord: i for i, coord in enumerate(novas_coords)}

        if self.subdivisoes == 1:
            novas_conec = self.conec
            novas_coords = self.coord
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

        # Atualizar as propriedades da estrutura
        self.coord = np.array(novas_coords, dtype=float)
        self.conec = np.array(novas_conec, dtype=int)
        self.num_elementos = int(len(self.conec))
        
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

    def CLOAD(self, esforcos_estrutura:dict):
        """
        Adiciona carga concentrada ao nó especificado, no formato [Px, Py, Pz].
        """
        for chave, forca in esforcos_estrutura.items():
            # Converter os esforços para arrays NumPy
            forca = np.array(forca)

            # Interpretar os nós (único ou múltiplo)
            nos = range(chave, chave + 1) if isinstance(chave, int) else chave

            for no in nos:
                # Adicionar os momentos concentrados ao no
                self.cargas_concentradas.append((no, forca.tolist()))    

    def MLOAD(self, esforcos_estrutura:dict):
        """
        Adiciona momento concentrado ao nó especificado, no formato [Mx, My, Mz].
        """
        for chave, momento in esforcos_estrutura.items():
            # Converter os momentos para arrays NumPy
            momento = np.array(momento)

            # Interpretar os nós (único ou múltiplo)
            nos = range(chave, chave + 1) if isinstance(chave, int) else chave

            for no in nos:
                # Adicionar os momentos concentrados ao no
                self.momentos_concentrados.append((no, momento.tolist()))


    def DLOAD(self, cargas_estrutura: dict):
        """
        Adiciona uma carga distribuída ao elemento, considerando subdivisões.
        """
        for chave, (carga_inicio, carga_fim) in cargas_estrutura.items():
            # Converter as cargas para arrays NumPy
            carga_inicio = np.array(carga_inicio)
            carga_fim = np.array(carga_fim)

            # Interpretar os elementos (único ou múltiplo)
            elementos = range(chave, chave + 1) if isinstance(chave, int) else chave

            for elem in elementos:
                # Salvar as cargas iniciais para fins de representação gráfica
                self.cargas_iniciais.append((elem, carga_inicio, carga_fim))

                if self.subdivisoes > 1:
                    # Fatores de interpolação para as cargas
                    fatores_inicio = np.linspace(0, 1, self.subdivisoes, endpoint=False)
                    fatores_fim = np.linspace(1 / self.subdivisoes, 1, self.subdivisoes)

                    # Interpolar as cargas nos subelementos
                    q_inicio_sub = (1 - fatores_inicio[:, None]) * carga_inicio + fatores_inicio[:, None] * carga_fim
                    q_fim_sub = (1 - fatores_fim[:, None]) * carga_inicio + fatores_fim[:, None] * carga_fim

                    # Gerar subelementos e armazenar as cargas subdivididas
                    cargas_subdivididas = [
                        (elem * self.subdivisoes + i, q_inicio_sub[i].tolist(), q_fim_sub[i].tolist())
                        for i in range(self.subdivisoes)
                    ]

                    # Atualizar as cargas distribuídas com as subdivisões
                    self.cargas_distribuidas.extend(cargas_subdivididas)

                else:
                    # Sem subdivisões: Adicionar a carga diretamente
                    self.cargas_distribuidas.append((elem, carga_inicio.tolist(), carga_fim.tolist()))
    
    def geometria(self, secoes: dict):
        """
        Define as seções da estrutura por intervalos de elementos.

        Args:
        - secoes_intervalos (dict): Mapeia intervalos (slice ou range) para dicionários de propriedades da seção.

        Exemplo:
        estrutura.geometria_por_intervalo({
            slice(0, 10): {"geometria": "retangular", "E": ..., "v": ..., "base": ..., "altura": ...},
            range(10, 20): {"geometria": "tubular", "E": ..., "v": ..., "raio_ext": ..., "raio_int": ...},
        })
        """
        # Inicializar a lista de seções
        self.secao_inicial = [{} for _ in range(len(self.conec_original))]
        # self.secao_inicial = [None] * len(self.conec_original)
        self.secoes = []

        # Iterar sobre os intervalos e definir as seções
        for intervalo, dados_secao in secoes.items():
            indices = range(*intervalo.indices(len(self.conec_original))) if isinstance(intervalo, slice) else list(intervalo)
            for i in indices:
                self.secao_inicial[i] = self._validar_secao(dados_secao)

        # Verificar se todas as seções foram definidas
        if any(s is None for s in self.secao_inicial):
            raise ValueError("Nem todos os elementos foram definidos.")
        
        # Aplicar as seções à estrutura subdividida
        for secao in self.secao_inicial:
            self.secoes.extend([secao] * self.subdivisoes)

    def _validar_secao(self, dados_secao):
        """
        Define as propriedades geométricas de uma seção transversal.

        Parâmetros:
        - tipo (str): O tipo da seção (e.g., "retangular", "circular", "tubular").
        - **kwargs: Parâmetros específicos para cada tipo de seção.
        """
        # Dicionário de validação de parâmetros para cada tipo de seção
        validacoes = {
            "retangular": {"base": float, "altura": float},
            "caixa": {"base": float, "altura": float, "espessura": float},
            "circular": {"raio": float},
            "tubular": {"raio_ext": float, "raio_int": float},
            "I": {"base": float, "altura": float, "espessura_flange": float, "espessura_alma": float},
            "T": {"base": float, "altura": float, "espessura_flange": float, "espessura_alma": float},
        }

        geometria = dados_secao.get("geometria")
        if geometria not in validacoes:
            raise ValueError(f"Tipo de seção '{geometria}' não reconhecido.")

        # Definir dados constitutivos da seção
        E = dados_secao.get("E")
        v = dados_secao.get("v")
        G = E / (2 * (1 + v)) if E and v else None

        # Inicializar os dados geométricos
        dimensoes = {
            "base": 0.0, "altura": 0.0, "raio": 0.0, "raio_ext": 0.0,
            "raio_int": 0.0, "espessura": 0.0, "espessura_flange": 0.0, "espessura_alma": 0.0,
        }

        # Validar e armazenar os parâmetros da seção
        for param, tipo_param in validacoes[geometria].items():
            valor = dados_secao.get(param)
            if valor is None:
                raise ValueError(f"Seção {geometria} precisa do parâmetro '{param}'.")
            if not isinstance(valor, tipo_param):
                raise ValueError(f"'{param}' deve ser do tipo {tipo_param.__name__}.")
            dimensoes[param] = valor

        return {"geometria": geometria, **dimensoes, "E": E, "v": v, "G": G}