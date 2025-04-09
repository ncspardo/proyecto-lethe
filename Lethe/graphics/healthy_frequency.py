import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('/home/gerardop/code/ncspardo/proyecto-lethe/Lethe/raw_data/bal_healthy.csv', header=None)

# Seleccionar filas específicas
df = df.iloc[[0, 4653, 6204, 7755]]

# Configuración de FFT
frecuencia_muestreo = 512  # Ajusta según tu dispositivo (ej. 100 Hz, 250 Hz, etc.)
n = 1024  # Número de puntos en la serie temporal (solo 1024 puntos, sin la columna 1025)

# Crear figura para los subgráficos
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Espectro de Frecuencia por Fase del Sueño - Healthy', fontsize=16)

# Diccionario de etiquetas de fases
fases = {
    0: "Vigilia (Wake)",
    1: "N1 (Transición)",
    2: "N2 (Sueño ligero)",
    3: "N3 (Sueño profundo)"
}

# Calcular y graficar FFT para cada fila (fase)
for i, (idx, row) in enumerate(df.iterrows()):
    # Obtener la señal EEG de la fila (convertir a array numpy)
    señal = row.iloc[:1024].values.astype(float)  # Tomar solo las primeras 1024 columnas
    
    # Calcular FFT
    fft_values = np.fft.fft(señal)
    fft_magnitudes = np.abs(fft_values)[:n//2]  # Solo frecuencias positivas
    frecuencias = np.fft.fftfreq(n, d=1/frecuencia_muestreo)[:n//2]
    
    # Seleccionar subgráfico
    ax = axs[i//2, i%2]
    
    # Graficar espectro
    ax.plot(frecuencias, fft_magnitudes, color='blue')
    ax.set_title(f'Fase: {fases[i]} (Etiqueta {i})')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud')
    ax.grid(True)
    ax.set_xlim(0, 100)  # Limitar a frecuencias relevantes para EEG (0-100 Hz)

plt.tight_layout()
plt.show()