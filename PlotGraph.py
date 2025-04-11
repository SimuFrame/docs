import numpy as np
import matplotlib.pyplot as plt

# Dados (viga)
# elementos = np.array([2, 4, 8, 16, 32, 64, 128])
# autor =  [12.845, 13.462, 13.607, 13.648, 13.659, 13.662, 13.663]
# abaqus = [13.171, 13.657, 13.654, 13.654, 13.654, 13.654, 13.654]
# ansys =  [11.959, 13.298, 13.624, 13.706, 13.726, 13.731, 13.732]
# rsa =    [14.630, 13.915, 13.736, 13.690, 13.676, 13.676, 13.676]

# Dados (pórtico espacial)
# elementos = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# autor =  [2.912, 2.915, 2.917, 2.918, 2.918, 2.918, 2.918, 2.918, 2.918, 2.918, 2.918]
# abaqus = [2.756, 2.880, 2.880, 2.895, 2.903, 2.903, 2.904, 2.904, 2.904, 2.904, 2.904]
# rsa =    [2.889, 2.892, 2.900, 2.901, 2.901, 2.903, 2.903, 2.903, 2.903, 2.903, 2.903]

# Tempo de execução
elementos = np.array([1, 2, 4, 8, 16, 32, 64])
tempo = [0.97, 1.15, 1.90, 4.35, 14.47, 57.27, 207.70]

# Criar o gráfico
plt.figure(figsize=(10, 6))

# Plotar as curvas
plt.plot(elementos, tempo, color='blue', linestyle='-', linewidth=2, marker='o', markersize=10, label='Tempo de execução')
# plt.plot(elementos, autor, label="SimuFrame", color="blue", linestyle="-", marker="o", markersize=10, linewidth=2)
# plt.plot(elementos, abaqus, label="Abaqus", color="green", linestyle="--", marker="s", markersize=10, linewidth=2)
# plt.plot(elementos, ansys, label="ANSYS", color="red", linestyle=":", marker="^", markersize=10, linewidth=2)
# plt.plot(elementos, rsa, label="RSA", color="purple", linestyle="-.", marker="d", markersize=10, linewidth=2)

# Configurações do gráfico
#plt.title("Comparação de Deslocamentos Máximos em Função do Número de Elementos", fontsize=14, fontweight="bold")
# plt.xlabel("Número de Elementos", fontsize=18)
# plt.ylabel("Deslocamento máximo (cm)", fontsize=18)
# plt.xscale("log", base=2)

plt.xlabel("Discretização dos elementos", fontsize=18)
plt.ylabel("Tempo de execução (s)", fontsize=18)

plt.xticks(elementos, labels=[str(e) for e in elementos], fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(loc="lower right", fontsize=20)

# Ajustes finais
plt.tight_layout()
plt.show()