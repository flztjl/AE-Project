#!/usr/bin/env python
# encoding: utf-8
'''
@author: Kolor
@license:
@contact: colorsu1922@163.com
@software: pycharm
@file: spec_sub.py
@time: 2020/12/21 23:44
@desc:
  Implements the basic power spectral subtraction algorithm [1].

  Usage:  specsub(noisyFile, outputFile)

         noisyFile - noisy speech file in .wav format
         outputFile - enhanced output file in .wav format

   Algorithm uses the first 5 frames to estimate the noise psd, and
   then uses a very simple VAD algorithm for updating the noise psd.
   Alternatively, the VAD in the file ss_rdc.m (this folder) can be used.

  References:
   [1] Berouti, M., Schwartz, M., and Makhoul, J. (1979). Enhancement of speech
       corrupted by acoustic noise. Proc. IEEE Int. Conf. Acoust., Speech,
       Signal Processing, 208-211.

 Author: Philipos C. Loizou

 Copyright (c) 2006 by Philipos C. Loizou
 $Revision: 0.0 $  $Date: 10/09/2006 $
-------------------------------------------------------------------------
'''
from typing import Type

import librosa
import numpy as np
import sys
import matplotlib.pyplot as plt
import soundfile as sf


def rh_spec_sub(infile, outfile):
    print(np.__version__)
    x, samp_rate = librosa.load(infile, sr=1000000, mono=True)
    # sys.exit("pause")
    # =============== Initialize variables ===============

    # Frame size in samples
    frame_len = int(np.floor(20 * samp_rate / 10000))
    if np.remainder(frame_len, 2) == 1:
        frame_len = frame_len + 1

    # window overlap in percent of frame size
    overlap_percent = 50
    len1 = int(np.floor(frame_len * overlap_percent / 100))
    len2 = int(frame_len - len1)

    # VAD threshold in dB SNR
    vad_thresh = 3
    # power exponent
    alpha = 2.0
    FLOOR = 0.002
    G = 0.9

    # define window
    win = np.hanning(frame_len+1)[0:frame_len]
    win_gain = len2 / np.sum(win)  # normalization gain for overlap+add with 50% overlap

    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence

    NFFT = int(2 * 2 ** next_pow2(frame_len))
    noise_mean = np.zeros(NFFT)
    j = 0
    for k in range(0, 5):
        windowed = np.multiply(win, x[j:j+frame_len])
        noise_mean = np.add(noise_mean, np.abs(np.fft.fft(windowed, NFFT)))
        j = j + frame_len

    noise_mu = noise_mean / 5

    # --- allocate memory and initialize various variables
    k = 0
    img = 1j  # np.sqrt(-1)
    x_old = np.zeros(len1)
    Nframs = int(np.floor(len(x)/len2) - 1)
    xfinal = np.zeros(Nframs * len2)

    # ===============================  Start Processing ==================================
    for n in range(0, Nframs):
        insign = np.multiply(win, x[k:k+frame_len])
        spec = np.fft.fft(insign, NFFT)
        sig = np.abs(spec)  # compute the magnitude

        # save the noisy phase information
        theta = np.angle(spec)

        SNR_seg = 10*np.log10(np.linalg.norm(sig, 2)**2 / np.linalg.norm(noise_mu, 2)**2)

        if alpha == 1.0:
            beta = berouti1(SNR_seg)
        else:
            beta = berouti(SNR_seg)

        # &&&&&&&&&&&&&&&
        sub_speech = sig ** alpha - beta * noise_mu ** alpha
        diffw = sub_speech - FLOOR * noise_mu ** alpha

        # floor negative component
        z = np.argwhere(diffw < 0)
        if len(z) is not 0:
            sub_speech[z] = FLOOR * noise_mu[z] ** alpha

        # --- implement a simple VAD detector - -------------
        if SNR_seg < vad_thresh:   # Update noise spectrum
            noise_temp = G * noise_mu ** alpha + (1 - G) * sig ** alpha
            noise_mu = noise_temp ** (1 / alpha)  # new noise spectrum

        # to ensure conjugate symmetry for real reconstruction
        sub_speech[int(NFFT/2) + 1: NFFT] = np.flipud(sub_speech[1: int(NFFT/2)])

        x_phase = (sub_speech**(1/alpha)) * (np.cos(theta) + img * (np.sin(theta)))

        # take the iFFT
        xi = np.real(np.fft.ifft(x_phase))
        # plt.plot(xi)
        # plt.show()
        # --- Overlap and add ---------------
        xfinal[k:k+len2] = x_old + xi[0:len1]
        x_old = xi[len1:frame_len]
        k = k + len2

    # write output
    sf.write(outfile, win_gain * xfinal, samp_rate, 'PCM_16')


def berouti1(snr):
    beta = 0
    if -5.0 <= snr <= 20:
        beta = 3 - snr * 2 / 20
    elif snr < -5.0:
        beta = 4
    elif snr > 20:
        beta = 1

    return beta


def berouti(snr):
    beta = 0
    if -5.0 <= snr <= 20:
        beta = 4 - snr * 3 / 20
    elif snr < -5.0:
        beta = 5
    elif snr > 20:
        beta = 1

    return beta


def next_pow2(n):
    return np.ceil(np.log2(np.abs(n)))


def load_audio(filename, trace=0):
    """
    load wav file using audioread.
    This is not available in python x,y.
    """
    data = np.array([])
    with audioread.audio_open(filename) as af:
        trace_n = af.channels
        if trace >= trace_n:
            print('number of traces in file is', trace_n)
            quit()
        nsamp = np.ceil(af.samplerate * af.duration)
        print(f"nsamp =%d" % nsamp)
        data = np.ascontiguousarray(np.zeros(nsamp, 1))
        index = 0
        for buffer in af:
            full_data = np.fromstring(buffer).reshape(-1, af.channels)
            n = full_data.shape[0]
            if index + n > len(data):
                n = len(data) - index
            if n > 0:
                data[index:index + n] = full_data[:n, trace]
                index += n
            else:
                break

    return af.samplerate, data, 'a.u.'
