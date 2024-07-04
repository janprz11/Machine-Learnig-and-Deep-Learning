import numpy as np
from tqdm import tqdm
import librosa
from scipy.signal import welch
import pywt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_features(wzbudzenia, type, fs = 50000, make_plots = False):
    widma = []
    mfccs_all = []
    stft_amplitudes = []
    stft_phases = []
    wavelet_trans_all = []
    fs = 50000
    longest = 0
    for signal in wzbudzenia:
        if(np.shape(signal)[0]>longest): longest = np.shape(signal)[0]

    for signal in tqdm(wzbudzenia):
        padded = np.pad(signal, (0,longest-np.shape(signal)[0]), mode="constant", constant_values=0)

        mfccs = librosa.feature.mfcc(y=padded, sr=fs, n_mfcc=30)
        mfccs_all.append(mfccs)

        d = librosa.stft(y = padded)
        stft_amplitudes.append(np.abs(d))
        stft_phases.append(np.angle(d))

        f, pxx = welch(padded, fs, nperseg=40000)
        widma.append(pxx)
        #print(f[np.argmax(pxx)])
    

        cA, cD1, cD2, cD3 = pywt.wavedec(padded, "haar", level=3)
        wavelet_trans_all.append(cD3)

    
    if(make_plots):
        print("Przygotywanie Grafik")
        # PSD
        plt.figure()
        plt.plot(f, pxx)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Signal power")
        plt.grid(True, "both")
        plt.xlim([0,12500])
        plt.tight_layout()
        plt.savefig(f"img\\PSD_{type}.png")

        # STFT
        plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(stft_amplitudes[4], ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(img, format="%+2.0f dB")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.xlim([0,3])
        plt.tight_layout()
        plt.savefig(f"img\\STFT{type}.png")

        # MFCC
        plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(mfccs_all[4], ref=np.max), y_axis='mel', x_axis='time')
        plt.colorbar(img, format="%+2.0f dB")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.xlim([0,3])
        plt.tight_layout()
        plt.savefig(f"img\\MFCC_{type}.png")

        # Wavelet
        plt.figure()
        plt.plot(wavelet_trans_all[4])
        plt.xlabel("Sample")
        plt.ylabel("Coefficient amplitude")
        plt.tight_layout()
        plt.xlim([5000, 10000])
        plt.grid(True, "both")
        plt.savefig(f"img\\Wavelet_3rd_degree_decomp_{type}.png")

    return np.asarray(widma), np.asarray(stft_amplitudes), np.asarray(mfccs_all), np.asarray(wavelet_trans_all)

def perform_PCA_reduction(data, type, variant, num_components, do_plots = True):
    # Obliczenie macierzy kowariancji
    cov_matrix = np.cov(data)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    pca = PCA(n_components=num_components)
    data_pca = pca.fit_transform(data)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues[::-1], 'o-')
    plt.xlabel('Eigenvector number')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.savefig(f"img\\PCA\\PCA_EIGENVALUES_{type}_{variant}.png")

    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual variance')
    plt.step(range(1, len(explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative variance')
    plt.xlabel('Eigenvector number')
    plt.ylabel('Percentage of explained variance')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"img\\PCA\\PCA_CUMSUM_{type}_{variant}.png")

    return data_pca