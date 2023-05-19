import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy.fft import fft, ifft
import torch


def moving_average(data, window_size):
    # Create an empty matrix to hold the smoothed data
    smoothed_data = np.empty_like(data)

    # Pad the data matrix with zeros on either side
    padded_data = np.pad(data, ((window_size // 2, window_size // 2), (0, 0)), mode='edge')

    # Apply the moving average to each column
    for i in range(data.shape[0]):
        smoothed_data[i, :] = np.mean(padded_data[i:i+window_size, :], axis=0)

    return smoothed_data


def gaussian_average(data, sigma):
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


def frequency_domain_filter(data, dt, cutoff_percentage):

    if type(data) is torch.Tensor:
        data = data.numpy()

    # Apply the FFT to each column to get the frequency components
    frequency_components = fft(data, axis=0)

    # Create a frequency index (time index -> frequency index)
    frequency_index = np.fft.fftfreq(data.shape[0], dt)
    frequency_index = np.abs(frequency_index)
    print('frequency_index: {}'.format(frequency_index))

    # Percentage of low and high cutoff frequencies
    threshold = cutoff_percentage * np.max(np.abs(frequency_index))
    low_cutoff = threshold
    high_cutoff = threshold

    # Create a mask for the frequencies we want to keep
    mask = (frequency_index >= low_cutoff) & (frequency_index <= high_cutoff)

    # Apply the mask to the frequency components
    filtered_frequency_components = frequency_components * mask[:, np.newaxis]

    # Apply the inverse FFT to get the filtered data
    filtered_data = np.real(ifft(filtered_frequency_components, axis=0))

    return filtered_data, frequency_index

