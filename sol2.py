from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from scipy import signal
from skimage.color import rgb2gray


def DFT(signal):
    """Compute the discrete Fourier Transform of the 1D array x"""
    N = len(signal)
    x = np.arange(N)
    u = x.reshape((N, 1))
    mat = np.exp(-2j*np.pi*u*x/N)
    dft_signal = np.dot(mat, signal)
    return dft_signal


def IDFT(fourier_signal):
    """Compute the discrete Fourier Transform of the 1D array x"""
    N = len(fourier_signal)
    x = np.arange(N)
    u = x.reshape((N, 1))
    mat = np.exp(2j*np.pi*u*x/N)
    mat /= N
    signal = np.dot(mat, fourier_signal)
    return signal


def DFT2(image):
    """
    Transform an image to its DFT representation using the previous functions
    :param image - matrix which represents an image:
    :return DFT representation of the image:
    """
    dim_im = np.ndim(image)
    if dim_im == 2:
        # (M.N) matrix as an image case, we will work on it
        im_orig = image.astype(np.complex128)
    else:
        # (M.N.1) matrix as an image case, we will divide it to 2d matrix and work on it
        im_orig = image[:, :, 0]
        im_orig = im_orig.astype(np.complex128)
    # DFT for rows
    im_orig = DFT(im_orig)
    # DFT for cols
    im_orig = DFT(im_orig.T)
    im_orig = im_orig.T
    if dim_im == 2:
        # no need to return to 3d so return 2d array.
        return im_orig
    else:
        # need to return to 3d.
        image = image.astype(np.complex128)
        image[:, :, 0] = im_orig
        return image


def IDFT2(fourier_image):
    """
    Transform a DFT representation of an image to its image representation using the previous functions
    :param fourier_image - DFT representation of an image:
    :return matrix which represents an image:
    """
    dim_im = np.ndim(fourier_image)
    if dim_im == 2:
        # (M.N) matrix as an image case, we will work on it
        im_orig = fourier_image.astype(np.complex128)
    else:
        # (M.N.1) matrix as an image case, we will divide it to 2d matrix and work on it
        im_orig = fourier_image[:, :, 0]
        im_orig = im_orig.astype(np.complex128)
    # IDFT for rows
    im_orig = IDFT(im_orig)
    # IDFT for cols
    im_orig = IDFT(im_orig.T)
    im_orig = im_orig.T
    if dim_im == 2:
        # no need to return to 3d so return 2d array.
        return im_orig
    else:
        # need to return to 3d.
        image = fourier_image.astype(np.complex128)
        image[:, :, 0] = im_orig
        return image


def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header
    :param filename - a string representing the path to a WAV file:
    :param ratio - a positive float64 representing the duration change:
    :return None:
    """
    rate, wav_rep = wavfile.read(filename)
    rate *= ratio
    wavfile.write("change_rate.wav", int(rate), wav_rep)


def change_samples(filename, ratio):
    """
    a fast forward function that changes the duration of an audio file by reducing the number of samples
    using Fourier. Also outputs the case where ratio = 1.
    :param filename -   a string representing the path to a WAV file:
    :param ratio -  a positive float64 representing the duration change:
    :return wav_rep - a 1D ndarray of dtype float64 representing the new sample points:
    """
    rate, wav_rep = wavfile.read(filename)
    resized_wav = resize(wav_rep, ratio)
    wavfile.write("change_samples.wav", rate, resized_wav.astype(np.int16))
    return resized_wav.astype(np.float64)


def resize(data, ratio):
    """
    resizes the given data array according to the ratio, if ratio > 1, then we need to get less samples, if
    ratio < 1 we need to get more samples. the case where ratio = 1 is dealt in change_samples function
    :param data - a 1D ndarray of dtype float64 or complex128 representing the original sample points:
    :param ratio - a positive float64 representing the duration change:
    :return final_wav - s a 1D ndarray of the dtype of data representing the new sample points:
    """
    # constants, apply DFT on data and shifting
    data_len = len(data)
    samples = int(data_len/ratio)
    clip_val = int(np.abs((data_len - samples) / 2))
    dft_wav = DFT(data)
    shift_dft_wave = np.fft.fftshift(dft_wav)
    # if ratio = 1 we dont need to do anything -return the data as is
    if ratio == 1:
        return data
    # if ratio > 1 then we need to cut the signal in the right places, then shifting back and IDFT
    if ratio > 1:
        shift_dft_wave = shift_dft_wave[clip_val:clip_val+samples]
        idft_wav = np.fft.ifftshift(shift_dft_wave)
        final_wav = IDFT(idft_wav)
    # if ratio < 1 then we need to pad in zeros the frequnices, then shift back and IDFT
    else:
        zeros = np.zeros(samples, dtype="complex128")
        zeros[clip_val:clip_val+int(data_len)] = shift_dft_wave
        final_wav = np.fft.ifftshift(zeros)
        final_wav = IDFT(final_wav)
    return np.real(final_wav)


def resize_spectrogram(data, ratio):
    """
    a function that speeds up a WAV file, without changing the pitch, using spectrogram scaling. the function uses the
    resize function above and apply it on each row of the spectogram, thus changing the number of spectrogram columns.
    :param data -  a 1D ndarray of dtype float64 representing the original sample points:
    :param ratio - a positive float64 representing the rate change of the WAV file:
    :return - the new sample points according to ratio with the same datatype as data:
    """
    spec = stft(data)
    spec_com = np.apply_along_axis(resize, 1, spec, ratio)
    return istft(spec_com)


def resize_vocoder(data, ratio):
    """
    a function that speedups a WAV file by phase vocoding its spectrogram and using spectrogram scaling.
    by using n use the supplied function phase_vocoder(spec, ratio), which scales the spectrogram spec
    by ratio and corrects the phases. the function then uses the resize function above and apply it on
    each row of the spectogram, thus changing the number of spectrogram columns.
    :param data:
    :param ratio:
    :return the new sample points according to ratio with the same datatype as data with corrected phase:
    """
    spec = stft(data)
    spec2 = phase_vocoder(spec, ratio)
    return istft(spec2)


def conv_der(im):
    """
    a function that computes the magnitude of image derivatives using convolution.
    :param im - nparray representing the image given:
    :return - the magnitude of the derivative of the image :
    """
    conv_x = signal.convolve2d(im, np.asarray([[0.5, 0, -0.5]]), "same")
    conv_y = signal.convolve2d(im, np.asarray([[0.5], [0], [-0.5]]), "same")
    magnitude = np.sqrt(np.abs(conv_x)**2 + np.abs(conv_y)**2)
    return magnitude


def fourier_der(im):
    """
    a function that computes the magnitude of the image derivatives using Fourier transform, uses the DFT2 and IDFT2
    functions given above. Based on the formulas we saw in class.
    :param im - a 2D matrix which represent an image:
    :return the magnitude of the derivative of the image:
    """
    fourier_im = DFT2(im)
    fourier_im = np.fft.fftshift(fourier_im)
    n, m = np.shape(fourier_im)
    #calculate the x derivative
    u = np.arange(-n/2, n/2)
    u = np.reshape(u, (n, 1))
    u_fourier_im = u*fourier_im
    u_fourier_im = np.fft.ifftshift(u_fourier_im)
    dx = IDFT2((2*np.pi*1j)/n*u_fourier_im)
    #calculate the Y derivative
    v = np.arange(-m/2, m/2)
    v_fourier_im = v*fourier_im
    v_fourier_im = np.fft.ifftshift(v_fourier_im)
    dy = IDFT2((2*np.pi*1j)/m*v_fourier_im)
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


"""helper function given in presubmit files"""


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def read_image(filename, representation):
    """
    reading the image
    :param filename - path to image:
    :param representation - int:
    :return picture in grayscale or rgb according to the input
    """
    im = imread(filename)
    if representation == 1:  # If the user specified they need grayscale image,
        if len(im.shape) == 3:  # AND the image is not grayscale yet
            im = rgb2gray(im)  # convert to grayscale (**Assuming its RGB and not a different format**)

    im_float = im.astype(np.float64)  # Convert the image type to one we can work with.

    if im_float.max() > 1:  # If image values are out of bound, normalize them.
        im_float = im_float / 255

    return im_float