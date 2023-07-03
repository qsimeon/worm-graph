import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter, find_peaks
from scipy import fftpack
import torch
from scipy.signal import butter, filtfilt
from scipy.stats import zscore


def moving_average_filter(data, window_size):
    # Create an empty matrix to hold the smoothed data
    smoothed_data = np.empty_like(data)

    # Pad the data matrix with zeros on either side
    padded_data = np.pad(data, ((window_size // 2, window_size // 2), (0, 0)), mode='edge')

    # Apply the moving average to each column
    for i in range(data.shape[0]):
        smoothed_data[i, :] = np.mean(padded_data[i:i+window_size, :], axis=0)

    return smoothed_data


def gaussian_average_filter(data, sigma):
    # Apply the Gaussian filter to each column
    smoothed_data = gaussian_filter1d(data, sigma, axis=0)

    return smoothed_data


def median_filter(data, window_size):
    # Apply the median filter to each column
    smoothed_data = medfilt(data, kernel_size=(window_size, 1))

    return smoothed_data


def savitzky_golay_filter(data, window_size, poly_order):
    # Apply the Savitzky-Golay filter to each column
    smoothed_data = savgol_filter(data, window_size, poly_order, axis=0)

    return smoothed_data


def frequency_filter(data, dt, percentage):

    sig_fft = fftpack.fft(data.detach().numpy(), axis=0) # FFT of the signal
    sample_freq = fftpack.fftfreq(data.size(0), d=dt) # Sample frequencies
    pos_mask = np.where(sample_freq >= 0)[0] # Positive mask frequencies

    # Exclude frequencies above the cutoff frequency
    freqs = sample_freq[pos_mask]
    cutoff_freq = (1 - percentage) * np.max(freqs)
    cutoff_mask = np.abs(sample_freq) > cutoff_freq

    high_freq_fft = sig_fft.copy()
    high_freq_fft[cutoff_mask, :] = 0

    filtered_sig = fftpack.ifft(high_freq_fft, axis=0)

    filtered_sig = torch.from_numpy(np.real(filtered_sig))

    return filtered_sig # Temporal signal


def temporal_filter(signal, cutoff_frequency, fs):
    nyquist_frequency = 0.5 * fs
    cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(1, cutoff, 'high')
    return filtfilt(b, a, signal)


def remove_artifacts(data, threshold):
    total_activity = data.sum(axis=1) 
    z_scores = zscore(total_activity)
    return data[np.abs(z_scores) < threshold, :]