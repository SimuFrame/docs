import matplotlib.pyplot as plt

# Dados de exemplo baseados no gráfico
discretizacao = [1, 2, 4, 8, 16, 32, 64, 128, 256]
tempo =         [0.67, 0.76, 0.93, 1.32, 2.15, 4.03, 7.44, 14.13, 27.58]
# elementos = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# simuframe = [2.764, 2.923, 2.889, 2.896, 2.903, 2.903, 2.902, 2.902, 2.902, 2.902, 2.902]
# abaqus =    [2.755, 2.880, 2.880, 2.895, 2.903, 2.904, 2.904, 2.904, 2.904, 2.904, 2.904]
# rsa =       [2.886, 2.889, 2.897, 2.900, 2.901, 2.903, 2.904, 2.904, 2.904, 2.904, 2.904]

# Criação do gráfico
plt.figure(figsize=(10, 6))

plt.plot(discretizacao, tempo, 'o-', markersize=10, color='blue', label='Tempo de execução')
# plt.plot(elementos, simuframe, 'o-', markersize=10, color='blue', label='SimuFrame')
# plt.plot(elementos, abaqus, 's--', markersize=10, color='green', label='Abaqus')
# plt.plot(elementos, rsa, 'd-.', markersize=10, color='purple', label='RSA')

# Eixos
plt.xlabel('Discretização da estrutura', fontsize=16)
# plt.xlabel('Número de Elementos', fontsize=16)
plt.ylabel('Tempo (s)', fontsize=16)
# plt.ylabel('Deslocamento máximo (cm)', fontsize=16)

# Grade e limites dos eixos
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
# plt.xlim(1, 20)
plt.xscale('log', base=2)
# plt.xticks(elementos, [str(e) for e in elementos], fontsize=14)
plt.xticks(discretizacao, [str(e) for e in discretizacao], fontsize=14)
plt.yticks(fontsize=14)

# Legenda
plt.legend(fontsize=18)

# Melhorar visual
plt.tight_layout()
plt.show()
