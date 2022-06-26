import numpy as np
from librosa import feature
from lws import lws


def melspectrogram(
    wave,
    sr,
    frame_period=10,
    overlap=4,
    n_mels=40,
    fmin=70,
    fmax=7680,
    eps=1.0e-5,
    **kwargs,
):
    fshift = int(sr * frame_period / 1000)
    fsize = fshift * overlap
    lws_processor = lws(fsize, fshift, **kwargs)

    X = lws_processor.stft(wave)
    A = np.abs(X)
    M = feature.melspectrogram(
        S=A.T,
        sr=sr,
        n_fft=fsize,
        hop_length=fshift,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=True,
    )

    return np.log(np.maximum(M, eps)).T


def mel_to_audio(
    M, sr, frame_period=10, overlap=4, n_mels=40, fmin=70, fmax=7680, **kwargs
):
    fshift = int(sr * frame_period / 1000)
    fsize = fshift * overlap
    lws_processor = lws(fsize, fshift, **kwargs)

    M = np.exp(M.T)
    A = feature.inverse.mel_to_stft(
        M, sr=sr, n_fft=fsize, fmin=fmin, fmax=fmax, htk=True
    )
    X = lws_processor.run_lws(A.T)
    wave = lws_processor.istft(X)
    return wave
