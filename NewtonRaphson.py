# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# Bibliotecas autorais
from IntegraçãoNumérica import analise_global
from Utilitários import expandir_dados


def verificar_convergencia(d, Δd, F, R, tol_forca=1e-6, tol_deslocamento=1e-6, epsilon=1e-12):
    """
    Verifica se o incremento de deslocamento converge com base nos critérios de força e deslocamento.

    Args:
    d (numpy array): Vetor de deslocamentos atuais.
    Δd (numpy array): Vetor de incrementos de deslocamento.
    F (numpy array): Vetor de forças atuais.
    R (numpy array): Vetor de forças residuais.
    tol_forca (float, optional): Tolerância para o critério de força.
    tol_deslocamento (float, optional): Tolerância para o critério de deslocamento.
    epsilon (float, optional): Valor epsilon para evitar divisão por zero.

    Returns:
    tuple: Um tuple contendo (convergencia, erro). 'convergencia' é um booleano que indica se o incremento
           convergiu. 'erro' é o erro máximo entre os critérios de força e deslocamento.
    """

    # Critério de força
    erro_forca = np.linalg.norm(R) / np.maximum(np.linalg.norm(F), epsilon)

    # Critério de deslocamento
    erro_deslocamento = np.linalg.norm(Δd) / np.maximum(np.linalg.norm(d), epsilon)

    # Verificar convergência
    convergencia = erro_forca < tol_forca and erro_deslocamento < tol_deslocamento

    return convergencia, max(erro_forca, erro_deslocamento)


def newton_raphson(F, KE, elementos, estrutura, numDOF, DOF, GLL, GLe, T, E, G, A, L, Ix, Iy, Iz, num_passos=5, max_iter=40, max_reducao=5):
    """
    Realiza o método de Newton-Raphson para resolver o problema de análise de estruturas
    não lineares. O método utiliza o esquema de integração da Quadratura de Gauss.
    O critério de convergência é verificado utilizando o critério de força e deslocamento.

    Args:
        Fr (numpy array): Vetor de forças externas.
        Ke (numpy array): Matriz de rigidez inicial.
        elementos (int): Número de elementos.
        estrutura (objeto): Instância da classe Estrutura.
        numDOF (int): Número de graus de liberdade.
        DOF (int): Número de graus de liberdade por nó.
        GLL (numpy array): Nós globais.
        GLe (numpy array): Elementos globais.
        T (numpy array): Matriz de transformação local-global.
        E (numpy array): Módulo de elasticidade.
        nu (numpy array): Coeficiente de Poisson.
        A (numpy array): Área da seção transversal.
        L (numpy array): Comprimento dos elementos.
        Ix (numpy array): Momento de inércia.
        num_passos (int, optional): Número de passos de carga. Defaults to 5.
        max_iter (int, optional): Número máximo de iterações internas. Defaults to 40.
        max_reducao (int, optional): Número máximo de reduções de passo. Defaults to 5.

    Returns:
        tuple: Um tuple contendo os deslocamentos e forças internas (dnl, fnl).
    """
    # Fatores de redução
    reducao_divergencia = 0.25
    reducao_convergencia_lenta = 0.5
    reducao_max_iter = 0.75

    # Inicializar os deslocamentos e forças internas
    d = np.zeros_like(F)
    Fint = np.zeros_like(F)

    # Matriz de rigidez tangente inicial (elástica)
    Kt = KE

    # Vetores para armazenar os erros
    erros = np.zeros(max_iter)

    incremento = 1            # Contador de incrementos
    reducao = 0               # Contador de reduções de passo
    ΔF = 1 / num_passos       # Tamanho inicial do passo de carga

    # Configurar o gráfico
    plt.ion()  # Ativar modo interativo
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Erro (log)")
    ax.set_yscale("log")

    # Criar linha para atualização
    linha_erro, = ax.plot([], [], "bo-", label="Erro do incremento")
    ax.axhline(y=1e-6, color='r', linestyle='--', label="Tolerância mínima")
    ax.legend()

    # Função para atualizar o gráfico
    def atualizar_grafico():
        ax.set_title(f"Convergência de Newton-Raphson: Incremento {incremento}/{num_passos}", fontdict={"fontsize": 14, "fontweight": "bold"})
        linha_erro.set_data(np.arange(1, iteracao), erros[:iteracao - 1])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.05)  # Pequena pausa para atualizar a tela

    while incremento <= num_passos:
        λ = ΔF * incremento         # Fator multiplicador da carga externa
        iteracao = 1                # Contador de iterações internas
        convergencia = False        # Inicializar critério de convergência

        # Salvar estado atual em caso de divergência
        d_backup, Fint_backup, Kt_backup = d.copy(), Fint.copy(), Kt.copy()

        while not convergencia and iteracao <= max_iter:
            # Calcular o vetor de forças residuais
            R = λ * F - Fint

            # Detectar divergência
            if np.linalg.norm(R) > 1e6:
                print(f"Divergência detectada na iteração {iteracao}...")
                convergencia = False
                break

            # Resolver o incremento de deslocamento Δd
            Δd = spsolve(Kt, R).reshape(-1, 1)

            # Atualizar deslocamentos
            d += Δd

            # Atualizar a matriz de rigidez tangente e o vetor de forças internas
            Kt, Fint, fe, de = analise_global(d, elementos, estrutura, numDOF, DOF, GLL, GLe, T, E, G, A, L, Ix, Iy, Iz, NLG=True)

            # Cálculo do critério de convergência
            convergencia, erros[iteracao - 1] = verificar_convergencia(d, Δd, λ * F, R)

            iteracao += 1

        # Atualizar gráfico
        atualizar_grafico()

        # Divergência ou falha na convergência
        if not convergencia or iteracao > max_iter:
            print(f"Incremento {incremento}: Não convergiu, reduzindo tamanho do passo...")
            d, Fint, Kt = d_backup, Fint_backup, Kt_backup  # Restaurar estado anterior
            reducao += 1

            # Aplicar fator de corte
            ΔF *= reducao_divergencia if not convergencia else reducao_max_iter if iteracao > max_iter else reducao_convergencia_lenta

            # Verifica o critério de passo mínimo
            if reducao > max_reducao:
                print("Erro: O tamanho do passo atingiu o valor mínimo permitido.")
                return d, de, fe # type: ignore

            # Corrigir o número do passo
            num_passos = int(1 / ΔF)
            continue

        incremento += 1  # Avança para o próximo incremento

    # Desativar o modo interativo e fechar o gráfico ao final
    plt.ioff()
    plt.close()

    # Expandir os deslocamentos e forças para 6 graus de liberdade
    dnl, fnl, _ = expandir_dados(elementos, estrutura.modelo, de, fe) # type: ignore

    return d, dnl, fnl
