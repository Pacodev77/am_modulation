# am_modulation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parámetros de la señal
Fs = 10000  # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1, Fs)  # 1 segundo de duración

# 1. Señal de mensaje (baja frecuencia)
f_m = 50  # Frecuencia de mensaje (Hz)
mensaje = np.sin(2 * np.pi * f_m * t)

# 2. Señal portadora (alta frecuencia)
f_c = 1000  # Frecuencia portadora (Hz)
portadora = np.cos(2 * np.pi * f_c * t)

# 3. Modulación AM
modulada = (1 + mensaje) * portadora

# 4. Graficar señales en el tiempo
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, mensaje, color="#00B3FF")  
plt.title("Señal de Mensaje (baja frecuencia)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.subplot(3, 1, 2)
plt.plot(t, portadora, color="#FFB700") 
plt.title("Portadora (alta frecuencia)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.subplot(3, 1, 3)
plt.plot(t, modulada, color="#79228B")  # verde oscuro
plt.title("Señal Modulada en AM")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# 5. Análisis de frecuencia usando FFT
def analizar_frecuencia(signal, Fs):
    N = len(signal)
    fft_result = fft(signal)
    fft_magnitude = np.abs(fft_result) / N
    freqs = fftfreq(N, 1 / Fs)
    return freqs[:N//2], fft_magnitude[:N//2]

# Obtener espectros
freq_mensaje, mag_mensaje = analizar_frecuencia(mensaje, Fs)
freq_portadora, mag_portadora = analizar_frecuencia(portadora, Fs)
freq_modulada, mag_modulada = analizar_frecuencia(modulada, Fs)

# 6. Graficar en dominio de la frecuencia
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(freq_mensaje, mag_mensaje, color='#FFA500')
plt.title("Espectro de la Señal de Mensaje")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(freq_portadora, mag_portadora, color='red')
plt.title("Espectro de la Portadora")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(freq_modulada, mag_modulada, color='#228B22')
plt.title("Espectro de la Señal Modulada en AM")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()

plt.tight_layout()
plt.show()

# Simulación de ruido blanco (AWGN)
ruido = np.random.normal(0, 0.5, size=modulada.shape)
modulada_con_ruido = modulada + ruido

# Graficar señal con ruido
plt.figure(figsize=(12, 4))
plt.plot(t, modulada_con_ruido, color='purple')
plt.title("Señal Modulada en AM con Ruido Blanco (AWGN)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.tight_layout()
plt.show()

# FFT de la señal con ruido
freq_ruido, mag_ruido = analizar_frecuencia(modulada_con_ruido, Fs)

plt.figure(figsize=(12, 4))
plt.plot(freq_ruido, mag_ruido, color='purple')
plt.title("Espectro de la Señal Modulada con Ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()
plt.tight_layout()
plt.show()

# Simulación de distorsión no lineal
alpha = 0.7  # Coeficiente de distorsión (ajustable)
modulada_distor = modulada + alpha * modulada**2

# Graficar señal distorsionada
plt.figure(figsize=(12, 4))
plt.plot(t, modulada_distor, color='darkorange')
plt.title("Señal Modulada en AM con Distorsión No Lineal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.tight_layout()
plt.show()

# FFT de la señal distorsionada
freq_distor, mag_distor = analizar_frecuencia(modulada_distor, Fs)

plt.figure(figsize=(12, 4))
plt.plot(freq_distor, mag_distor, color='darkorange')
plt.title("Espectro de la Señal Distorsionada")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()
plt.tight_layout()
plt.show()

# 9. Simulación de atenuación
factor_atenuacion = 0.3  # Reducimos la amplitud al 30%
modulada_atenuada = factor_atenuacion * modulada

# Graficar señal atenuada
plt.figure(figsize=(12, 4))
plt.plot(t, modulada_atenuada, color='teal')
plt.title("Señal Modulada en AM con Atenuación")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.tight_layout()
plt.show()

# FFT de la señal atenuada
freq_atenua, mag_atenua = analizar_frecuencia(modulada_atenuada, Fs)

plt.figure(figsize=(12, 4))
plt.plot(freq_atenua, mag_atenua, color='teal')
plt.title("Espectro de la Señal Atenuada")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()
plt.tight_layout()
plt.show()

