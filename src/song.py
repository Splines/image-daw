import os
from PIL import Image
import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
from matplotlib import pyplot as plt

SR = 5000 # sample frequency in Hz
NAME = 'ps-song'

image = Image.open(os.path.join('./data/', NAME + '.png')).convert('L') # luminance mode (greyscale)
width,height = image.size
print(f'width: {width}')
print(f'height: {height}')

image_data = np.array(image, dtype=np.float32)
print(f'image_data shape: {image_data.shape}') # height x width

# index by frequency, get list of amplitudes of that frequency per time
# freq_data = np.zeros((width, height), dtype=np.float64)
print('Processing image...')
freq_data = image_data / 255.0
freq_data = freq_data[:, ::-1] # reverse rows (due to convolution)
print(f'freq_data shape: {freq_data.shape}')
print('...Done processing image')

# Calculate nperseg and noverlap, such that total song length is SONG_LENGTH seconds
NPERSEGMENT = int(0.8 * SR) # k
WINDOW=signal.windows.kaiser(NPERSEGMENT, 5, sym=False)
T = 20 # song length in seconds
NOVERLAP = int(SR/(1-width) * (T - (NPERSEGMENT * width) / SR)) # r
# NOVERLAP = NPERSEGMENT / 2

print(f'NPERSEGMENT: {NPERSEGMENT}')
print(f'NOVERLAP: {NOVERLAP}')

# Fourier
time,values = signal.istft(freq_data, fs=SR, input_onesided=True, nperseg=NPERSEGMENT, noverlap=NOVERLAP, window=WINDOW)

# Save file
audio = np.real(values).astype(np.float32)
print(f'audio: {audio}')
print(f'Min audio: {np.min(audio)}')
print(f'Max audio: {np.max(audio)}')
print(f'Type audio: {type(audio[0])}')
wave.write(os.path.join('./data/', NAME + '.wav'), SR, audio)

# plt.figure(figsize=(20,15))
# plt.plot(time, values)
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()

# 💠
print('back to frequencies')
f, t, Zxx = signal.stft(audio, fs=SR, return_onesided=True, nperseg=NPERSEGMENT, noverlap=NOVERLAP, window=WINDOW)
# print(f'f: {f}')
# print(f't: {t}')
# print(f'Zxx: {Zxx}')
min = np.min(Zxx)
max = np.max(Zxx)
print(f'Min Zxx: {np.min(Zxx)}')
print(f'Max Zxx: {np.max(Zxx)}')

plt.figure(figsize=(20, 15))
plt.pcolormesh(t, f, np.real(Zxx), vmin=0.0, vmax=np.real(max), cmap='magma')
plt.colorbar(label="STFT magnitude")
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

