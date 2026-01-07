import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from scipy.stats import alpha

FIGSIZE = (15, 10)

file = "/Volumes/MacDrive_Ex/Code/machine learning/w3s_ml/src/digital-signal-processing/learning/musikalkemist/deep_learning_for_audio/11_preprocessing_audio_data_for_deep_learning/blues.wav"

# Load audio
signal, sample_rate = librosa.load(file, sr=22050)



# WAVEFORM
plt.figure(figsize=FIGSIZE)
librosa.display.waveshow(signal, sr=sample_rate, alpha=0.4)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")



# FFT
fft = np.fft.fft(signal)

# Spectrum
spectrum = np.abs(fft)

# Create frequency variable
f = np.linspace(
    0,
    sample_rate,
    len(spectrum)
)


# Take a half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum) / 2)]
left_f = f[:int(len(spectrum) / 2)]


# PLot Spectrum
plt.figure(figsize=FIGSIZE)
plt.plot(left_f, left_spectrum,  alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Spectrum")


# STFT -> spectrogram
hop_length = 512 # in num. of samples
n_fft = 2048 # window in num. of samples


# Calculate duration hop length and window in seconds
hop_length_duration = float(hop_length) / sample_rate
n_fft_duration = float(n_fft) / sample_rate

print("STFT hop length duration is: {}".format(hop_length_duration))
print("STFT window duration is: {}".format(n_fft_duration))


# Perform STFT
stft = librosa.stft(
    signal,
    n_fft=n_fft,
    hop_length=hop_length
)

# Calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)


# Display spectrogram
plt.figure(figsize=FIGSIZE)
librosa.display.specshow(
    spectrogram,
    sr=sample_rate,
    hop_length=hop_length
)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")



# Apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)


plt.figure(figsize=FIGSIZE)
librosa.display.specshow(
    log_spectrogram,
    sr=sample_rate,
    hop_length=hop_length
)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.colorbar(format="%+2.0f dB")
plt.title("Log Spectrogram")




# MFCCs (13 MFCCs)
MFCCs = librosa.feature.mfcc(
    y=signal,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mfcc=13
)

plt.figure(figsize=FIGSIZE)
librosa.display.specshow(
    MFCCs,
    sr=sample_rate,
    hop_length=hop_length
)
plt.xlabel("Time")
plt.ylabel("MFCCs coefficients")
plt.colorbar(format="%+2.0f dB")
plt.title("MFCCs")


plt.show()