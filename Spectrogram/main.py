# This is a python script to generate a spectrogram for an input file
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def Spectrogram(x, Fs, overlap, step, N_start, NNstep):
    N_step = int(NNstep * overlap)
    N_rec = int(np.floor(len(x)/(N_step)))
    w = np.hanning(step)
    spec = np.empty([int(step / 2), int(N_rec - 1)])
    for m in range(int(N_rec - 2)):
        N_stop = int(N_start + (step))
        if N_stop > len(x):
            break
        yy = w * x[N_start:N_stop]
        Sxx = 2 * (df * (np.abs(np.fft.fft(yy)) ** 2))
        Gxx = Sxx[0:int(step / 2)].copy()
        Gxx[0] /= 2
        Gxx /= np.max(np.abs(Gxx))
        spec[:, m] = Gxx
        N_start += N_step
        N_stop += N_step

    spec = np.flip(spec.copy(), axis=0)

    # plot initialization
    tVec = np.linspace(0, dt, int((N * dt)), endpoint=False)
    fVec = np.linspace(0, df, int(Fs / 2), endpoint=False)

    fig, ax = plt.subplots()
    plt.imshow(spec, extent=[0, int(N * dt), 0, int(Fs / 2)], aspect='auto')
    plt.xlabel("Time(s)")
    plt.ylabel("Frequency (Hz)")

    plt.show()


# main script run
if __name__ == '__main__':
    Fs, x = wav.read('./V0_LFM15K.wav')
    dt = 1 / Fs
    N = len(x)
    df = 1 / (dt * N)
    t = 1 / df

    overlap = .25
    step = overlap * 512
    N_start = 0
    NNstep = 512
    y = Spectrogram(x,Fs,overlap,step,N_start,NNstep)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
