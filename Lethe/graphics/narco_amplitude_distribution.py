import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('/home/gerardop/code/ncspardo/proyecto-lethe/Lethe/raw_data/bal_narco.csv', header=None)

# Seleccionar filas específicas
df = df.iloc[[0, 4779, 6372, 7965]]

# Configuración de las gráficas
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribución de Amplitud EEG por Fase del Sueño - Narco', fontsize=16)

# Diccionario de etiquetas de fases
fases = {
    0: "Vigilia (Wake)",
    1: "N1 (Transición)",
    2: "N2 (Sueño ligero)",
    3: "N3 (Sueño profundo)"
}

# Graficar la distribución (histograma) de las señales EEG para cada fila (fase)
for i, (idx, row) in enumerate(df.iterrows()):
    # Obtener la señal EEG de la fila (convertir a array numpy), solo las primeras 1024 columnas
    señal = row.iloc[:1024].values.astype(float)
    
    # Seleccionar subgráfico
    ax = axs[i//2, i%2]
    
    # Graficar el histograma de la señal EEG
    ax.hist(señal, bins=50, color='red', alpha=0.7)  # 'bins' controla el número de barras
    ax.set_title(f'Distribución de Amplitud EEG - Fase: {fases[i]} (Etiqueta {i})')
    ax.set_xlabel('Amplitud (microVolts)')
    ax.set_ylabel('Frecuencia')
    ax.grid(True)

# Ajustar el layout de los gráficos
plt.tight_layout()
plt.show()