"""Synthetic-spectrum distortion augmentation, applied on-the-fly during training and testing."""

"""Synthetic-spectrum distortion augmentation, applied on-the-fly during training and testing."""

from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve, hilbert

DataAugmentationPartial = Callable[[np.ndarray], np.ndarray]


def add_phase_distortion(
        spectrum: np.ndarray,
        ppm_scale: np.ndarray,
        ppm_right: float,
        ppm_left: float,
        phase_0_custom: float = None,
        phase_1_custom: float = None,
        use_custom_values: bool = False,
        phase_0_bound: float = 8.0,
        phase_1_factor: float = 8.0
):
    if use_custom_values and phase_0_custom is not None and phase_1_custom is not None:
        phase_0 = phase_0_custom
        phase_1 = phase_1_custom
    else:
        phase_0 = np.random.uniform(-phase_0_bound, phase_0_bound)
        ppm_width = ppm_left - ppm_right
        phase_1 = np.random.uniform(-phase_1_factor / ppm_width, +phase_1_factor / ppm_width)

    spectrum *= np.exp(1j * np.pi / 180 * (phase_0 + phase_1 * ppm_scale))
    return spectrum


def add_noise(
        spectrum: np.ndarray,
        custom_SNR: float = None,
        use_custom_values: bool = False,
        snr_lower_bound: float = 2.0,
        snr_upper_bound: float = 5.0
):
    if use_custom_values and custom_SNR is not None:
        # Use the provided custom SNR
        sampled_SNR = custom_SNR
    else:
        # Sample SNR within the provided bounds
        sampled_SNR = 10.0 ** np.random.uniform(snr_lower_bound, snr_upper_bound)

    # Compute the noise level based on the SNR
    max_noise_level = max(np.real(spectrum)) / (sampled_SNR * 2)

    # Add noise to the spectrum using the computed noise level
    spectrum += np.random.normal(scale=max_noise_level, size=spectrum.shape[0])

    # Return the modified spectrum and the sampled (or custom) SNR
    return spectrum, sampled_SNR

def add_13C_satellites_with_variability(
        spectrum,
        j_coupling_min: float = 40.0,
        j_coupling_max: float = 220.0,
        satellite_intensity_min: float = 0.005,
        satellite_intensity_max: float = 0.015,
        points_per_Hz: float = 5.12
):
    """Add 13C satellite peaks with variability in shift magnitude, truncating at the edges."""
    size = len(spectrum)

    # Calculating shift magnitude in points
    j_coupling_constant_Hz = np.random.uniform(j_coupling_min, j_coupling_max)
    base_shift_magnitude = int(j_coupling_constant_Hz * points_per_Hz)
    shift_magnitude = base_shift_magnitude // 2  # Half of the J-coupling constant

    # Create empty arrays for satellite peaks
    shifted_spectrum_left = np.zeros_like(spectrum)
    shifted_spectrum_right = np.zeros_like(spectrum)

    # Scale the intensity of satellite peaks
    satellite_intensity = np.random.uniform(satellite_intensity_min, satellite_intensity_max)

    # Adding satellite peaks with truncation at the edges
    for i in range(size):
        if spectrum[i] > 0:
            # Left satellite peak
            left_index = i - shift_magnitude
            if 0 <= left_index < size:  # Check if within spectrum range
                shifted_spectrum_left[left_index] = spectrum[i] * satellite_intensity

            # Right satellite peak
            right_index = i + shift_magnitude
            if 0 <= right_index < size:  # Check if within spectrum range
                shifted_spectrum_right[right_index] = spectrum[i] * satellite_intensity

    # Combine the original and satellite spectra
    augmented_spectrum = spectrum + shifted_spectrum_left + shifted_spectrum_right
    return augmented_spectrum


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a Gaussian kernel."""
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()  # Normalize
    return g


def sample_sigma(min_val=0.0, max_val=15.0, lambda_param=2., custom_value: float = None,
                 use_custom_values: bool = False):
    if use_custom_values and custom_value is not None:
        sigma = custom_value
    else:
        sampled_value = -np.log(1 - np.random.rand()) / lambda_param
        sigma = min_val + (max_val - min_val) * (sampled_value / 10.0)

    return sigma


def add_line_broadening(spectrum: np.ndarray, custom_sigma: float = None,
                        use_custom_values: bool = False) -> np.ndarray:
    sigma = sample_sigma(custom_value=custom_sigma, use_custom_values=use_custom_values)
    if sigma == 0.0:
        return spectrum
    else:
        kernel_size = int(6 * sigma) | 1  # Ensure the kernel size is odd
        if kernel_size < 3:
            kernel_size = 3

        kernel = gaussian_kernel(kernel_size, sigma)
        broadened_spectrum = fftconvolve(spectrum, kernel, mode='same').astype(np.complex128)

        return broadened_spectrum


def add_baseline_distortion(
        spectrum: np.ndarray,
        ppm_scale: np.ndarray,
        ppm_right: float,
        ppm_left: float,
        sino: float,
        min_peak: float = 1.0,
        base_scale: float = .5,
        custom_base_left: float = None,
        custom_base_right: float = None,
        use_custom_values: bool = False
):
    if use_custom_values and custom_base_left is not None and custom_base_right is not None:
        base_left = custom_base_left
        base_right = custom_base_right
    else:
        base_level = min_peak / sino * base_scale
        base_left = np.random.uniform(-base_level, base_level)
        base_right = np.random.uniform(-base_level, base_level)

    position_in_scale = (ppm_scale - ppm_left) / (ppm_right - ppm_left)
    spectrum += position_in_scale * base_right + (1 - position_in_scale) * base_left
    return spectrum


def find_nearest(array, value):
    """Find the index of the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def measure_fwhm(x, y):
    """Measure the FWHM of a peak in the spectrum."""
    # Find the peak maximum
    ymax = np.max(y)
    xmax = x[y.argmax()]

    # Calculate half maximum
    half_max = ymax / 2

    # Find indices of the half maximum points
    left_idx = find_nearest(y[:y.argmax()], half_max)
    right_idx = find_nearest(y[y.argmax():], half_max) + y.argmax()  # adjust for slice

    # Measure the FWHM
    fwhm = x[right_idx] - x[left_idx]

    return fwhm, xmax, ymax, x[left_idx], x[right_idx]


def add_shim_distortions(
        spectrum: np.ndarray,
        sweep_width: float = 500000,
        upsampling_factor: int = 4,
        Z1LIM: float = 20,
        X1LIM: float = 20.,
        Y1LIM: float = 20.0,
        Z2LIM: float = 5.0,
        plotting_enabled: bool = False  # Add plotting flag with default value
):
    # Lazy import: ShimSim is GPL-derived (adapted from SHIMpanzee, GNU GPL). Importing it
    # inside the only function that uses it keeps ``import moldetr.dataloader.data_augmentation``
    # -- and thus ``moldetr.distort`` (which wraps only the Apache-2.0 add_* effects) -- free of
    # any transitive GPL import. See the licensing-boundary note in moldetr/distort.py.
    from moldetr.dataloader.shimming import ShimSim

    # Original Spectrum
    if plotting_enabled:
        # Assuming 'measure_fwhm' function is defined and will be used here to measure FWHM and peak characteristics
        fwhm, peak_x, peak_y, left_half_max_x, right_half_max_x = measure_fwhm(np.arange(spectrum.size),
                                                                               np.real(spectrum))
        print(f"FWHM: {fwhm}, Peak at: {peak_x}, Peak height: {peak_y}")
        print(f"Left half-max at: {left_half_max_x}, Right half-max at: {right_half_max_x}")
        plt.figure(figsize=(30, 10))
        plt.plot(np.real(spectrum), label='Real Part', color='blue')
        plt.plot(np.imag(spectrum), label='Imaginary Part', color='red', linestyle='--')
        plt.title('Original Spectrum')
        plt.legend()
        plt.show()

    # Inverse Fourier Transform to get fid
    fid = np.fft.ifft(np.fft.ifftshift(spectrum))
    if plotting_enabled:
        plt.figure(figsize=(30, 10))
        plt.plot(np.real(fid), label='Real Part', color='blue')
        plt.plot(np.imag(fid), label='Imaginary Part', color='red', linestyle='--')
        plt.title('FID after IFFT')
        plt.legend()
        plt.show()

    # Upsampling
    fid_upsampled = np.zeros(fid.size * upsampling_factor, dtype=np.complex128)
    fid_upsampled[:fid.size // 2] = fid[:fid.size // 2]
    fid_upsampled[-fid.size // 2:] = fid[-fid.size // 2:]
    fid_upsampled[0] = fid_upsampled[0] * 0.5
    length_upsampled = len(fid_upsampled)
    if plotting_enabled:
        plt.figure(figsize=(30, 10))
        plt.plot(np.real(fid_upsampled), label='Real Part', color='blue')
        plt.title('FID Upsampled')
        plt.legend()
        plt.show()

        # ShimSim and simulate
    sim = ShimSim(npoints=length_upsampled, sw=sweep_width)
    sim.startGame()

    distortion_value = np.random.uniform(-1, 1, 4)
    sim.simulate(z1=distortion_value[0] * Z1LIM, z2=distortion_value[1] * Z2LIM, x1=distortion_value[2] * X1LIM,
                 y1=distortion_value[3] * Y1LIM)
    sample = {'shim spectrum': sim.spectrum, 'trial': distortion_value,
              'fwhm': sim.fwhm * (length_upsampled / sweep_width),
              'peak_max': sim.peak_max, 'fidSurface': sim.fidSurface}

    shim_spectrum = sample['shim spectrum']

    if plotting_enabled:
        # Original Shim Spectrum zoomed in around the peak
        plt.figure(figsize=(30, 10))
        plt.plot(shim_spectrum, label='Shim Spectrum')
        plt.xlim(shim_spectrum.size // 2 - 100, shim_spectrum.size // 2 + 100)
        plt.title('Shim Spectrum Zoomed In')
        plt.legend()
        plt.show()

    # Apply Hilbert transform and take the complex conjugate
    complex_shim_spectrum = np.conj(hilbert(shim_spectrum))
    if plotting_enabled:
        # Complex Shim Spectrum zoomed in around the peak
        plt.figure(figsize=(30, 10))
        plt.plot(np.real(complex_shim_spectrum), label='Real Part')
        plt.plot(np.imag(complex_shim_spectrum), label='Imaginary Part', linestyle='--')
        plt.xlim(len(complex_shim_spectrum) // 2 - 100, len(complex_shim_spectrum) // 2 + 100)
        plt.title('Complex Shim Spectrum (Magnitude) Zoomed In')
        plt.legend()
        plt.show()

    # Perform IFFT and IFFT shift
    complex_shim_fid = np.fft.ifft(np.fft.ifftshift(complex_shim_spectrum))
    complex_shim_fid[0] = complex_shim_fid[0] * 0.5

    # Multiply with fid_upsampled
    multiplied_time_signal = fid_upsampled * complex_shim_fid
    if plotting_enabled:
        # Original multiplied time-domain signal
        plt.figure(figsize=(30, 10))
        plt.plot(np.real(multiplied_time_signal), label='Real Part - Original')
        plt.plot(np.imag(multiplied_time_signal), label='Imaginary Part - Original', linestyle='--')
        plt.title('Original Multiplied Time-domain Signal')
        plt.legend()
        plt.show()

    # Apply FFT to the multiplied time-domain FID to get the spectrum
    full_spectrum = np.fft.fftshift(np.fft.fft(multiplied_time_signal))

    # "Downsample" by selecting every fourth point from the full spectrum
    # This is actually decimating the spectrum data for visualization
    decimated_spectrum = full_spectrum[::4]
    if plotting_enabled:
        # Plot the magnitude of the full and decimated spectra for comparison
        plt.figure(figsize=(30, 10))
        plt.plot(np.real(spectrum) / np.max(np.real(spectrum)), label='Real Part of Original Spectrum')
        plt.plot(np.real(decimated_spectrum) / np.max(np.real(decimated_spectrum)),
                 label='Real Part of Decimated Spectrum', linestyle='--')
        plt.title('Comparison of Full and Decimated Spectra')
        plt.legend()
        plt.show()

        # Optional: Plot real and imaginary parts of the decimated spectrum
        plt.figure(figsize=(30, 10))
        plt.plot(np.real(decimated_spectrum), label='Real Part - Decimated')
        plt.plot(np.imag(decimated_spectrum), label='Imaginary Part - Decimated', linestyle='--')
        plt.title('Decimated Spectrum (Real and Imaginary Parts)')
        plt.legend()
        plt.show()

        # Assuming 'measure_fwhm' function is defined and will be used here to measure FWHM and peak characteristics
        fwhm, peak_x, peak_y, left_half_max_x, right_half_max_x = measure_fwhm(np.arange(decimated_spectrum.size),
                                                                               np.real(decimated_spectrum))
        print(f"FWHM: {fwhm}, Peak at: {peak_x}, Peak height: {peak_y}")
        print(f"Left half-max at: {left_half_max_x}, Right half-max at: {right_half_max_x}")

    # Reset simulation for the next sweep width iteration
    sim.reset()

    return decimated_spectrum


def augment_distortions(
    spectrum: np.ndarray,
    ppm_right: float,
    ppm_left: float,
    zoom_ppm_min: float = 0.95,
    zoom_ppm_max: float = 1.05,
    plotting: bool = False,
    # Custom values and flags for deterministic distortions
    use_custom_values: bool = False,
    phase_0_custom: float = None,
    phase_1_custom: float = None,
    base_left_custom: float = None,
    base_right_custom: float = None,
    custom_snr_value: float = None,
    sigma_custom: float = None


):
    scale_array = np.linspace(ppm_right, ppm_left, spectrum.shape[0], dtype=np.complex128)



    if plotting:
        # Original spectrum visualization
        plt.figure(figsize=(30, 10))
        plt.plot(scale_array, np.real(spectrum), "b-", linewidth=1.0, label="Original Spectrum")
        plt.title("Original Spectrum")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Intensity")
        plt.show()

    # Add 13C satellites with variability
    # Assuming this function is compatible with the new scheme
    spectrum = add_13C_satellites_with_variability(spectrum)

    if plotting:
        # Plotting the spectrum
        plt.figure(figsize=(30,10))

        # Normal plot
        plt.plot(scale_array, spectrum, label=r'Spectrum with Variable \(^{13}C\) Satellites', linestyle='--')
        plt.title(r'Spectrum with Variable \(^{13}C\) Satellites')
        plt.show()
        plt.savefig('spectrum_with_variable_13C_satellites.png')

    # Line broadening
    # spectrum = add_line_broadening(spectrum, custom_sigma=sigma_custom, use_custom_values=use_custom_values)
    # # Plot FID (time domain) and
    #
    toss_coin = 0.99 #np.random.uniform(0, 1)

    if toss_coin < 0.5:
        sweep_width = np.random.uniform(50000, 1000000)
        # sweep_width = sigma_custom
        # print("sweep_width: ",sweep_width)
        spectrum=add_shim_distortions(spectrum,sweep_width=sweep_width)
    elif toss_coin < 0.95 and toss_coin >= .6:
        # Line broadening
        spectrum = add_line_broadening(spectrum, custom_sigma=sigma_custom, use_custom_values=use_custom_values)
    else:
        pass

    #normalize the spectrum
    spectrum = spectrum/np.max(np.real(spectrum))


    if plotting:
        plt.figure(figsize=(30, 10))
        plt.plot(scale_array, np.real(spectrum), "r-", linewidth=1.0, label="Spectrum after Line Broadening")
        plt.title("Spectrum after Line Broadening")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Intensity")
        plt.show()


    # Phase distortion
    spectrum = add_phase_distortion(spectrum, scale_array, ppm_right, ppm_left, phase_0_custom, phase_1_custom,
                                    use_custom_values)
    if plotting:
        plt.figure(figsize=(30, 10))
        plt.plot(scale_array, np.real(spectrum), "g-", linewidth=1.0, label="Spectrum after Phase Distortion")
        plt.title("Spectrum after Phase Distortion")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Intensity")
        plt.show()

    # Noise and baseline distortion

    spectrum, snr = add_noise(spectrum, custom_SNR=custom_snr_value, use_custom_values=use_custom_values)
    spectrum = add_baseline_distortion(spectrum, scale_array, ppm_right, ppm_left, snr, custom_base_left=base_left_custom, custom_base_right=base_right_custom, use_custom_values=use_custom_values)
    if plotting:
        plt.figure(figsize=(30, 10))
        plt.plot(scale_array, np.real(spectrum), "m-", linewidth=1.0, label="Spectrum after Noise and Baseline Distortion")
        plt.title("Spectrum after Noise and Baseline Distortion")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Intensity")
        plt.show()


    # Zoom in on a specific region
    zoom_region_mask = (scale_array >= zoom_ppm_min) & (scale_array <= zoom_ppm_max)
    zoom_scale_array = scale_array[zoom_region_mask]
    zoom_spectrum = np.real(spectrum)[zoom_region_mask]
    if plotting:
        plt.figure(figsize=(30, 10))
        plt.plot(zoom_scale_array, zoom_spectrum, "y-", linewidth=1.0, label="Zoomed-in Spectrum")
        plt.title(f"Zoomed-in Spectrum (ppm: {zoom_ppm_min}-{zoom_ppm_max})")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()
    return spectrum
