import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('/home/gerardop/code/ncspardo/proyecto-lethe/Lethe/raw_data/bal_healthy.csv', header=None)

# Seleccionar filas específicas
df = df.iloc[[0, 4653, 6204, 7755]]

# Configuración de las gráficas
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Señal EEG por Fase del Sueño - Healthy', fontsize=16)

# Diccionario de etiquetas de fases
fases = {
    0: "Vigilia (Wake)",
    1: "N1 (Transición)",
    2: "N2 (Sueño ligero)",
    3: "N3 (Sueño profundo)"
}

# Graficar las señales EEG para cada fila (fase)
for i, (idx, row) in enumerate(df.iterrows()):
    # Obtener la señal EEG de la fila (convertir a array numpy), solo las primeras 1024 columnas
    señal = row.iloc[:1024].values.astype(float)
    
    # Seleccionar subgráfico
    ax = axs[i//2, i%2]
    
    # Graficar la señal EEG como dispersión (scatter plot)
    ax.scatter(range(1024), señal, color='blue', s=10)  # 's=10' es el tamaño de los puntos
    ax.set_title(f'Señal EEG - Fase: {fases[i]} (Etiqueta {i})')
    ax.set_xlabel('Tiempo (muestras)')
    ax.set_ylabel('Amplitud (microVolts)')
    ax.grid(True)

# Ajustar el layout de los gráficos
plt.tight_layout()
plt.show()