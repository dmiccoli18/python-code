import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt

def conversionFactor(calibration, frequency, V): # add V
    conv = calibration.to_numpy()
    conv = conv[:, 0:3]
    freq = conv[:, 0]
    VtoPa = conv[:, 2]

    index = 0
    while index < len(freq):
        if frequency == freq[index]:
            print(index)
            break
        else:
            index = index + 1

    convFactor = VtoPa[index]
    print(1 / convFactor)

    Pa = (1/convFactor) * V
    return convFactor, Pa # (returns voltage to pressure conversion)


def FDAValues(Pa,t,f,z):
    # PII
    dt = t[1]-t[0]
    Pa_s = np.square(Pa)
    integral = sum(dt * Pa_s)

    PII = integral / z
    print( "PII: " + str(PII) + " W*s/m^2")

    # MI (< 1.9)
    p_neg = abs(min(Pa)) * (10**-6)
    MI = p_neg/np.sqrt(f)
    print("Mechanical Index: "+ str(MI))

    # I_sppa (190 W/cm^2)
    PD = t[-1]-t[0]
    I_sppa = PII/PD
    I_sppa = I_sppa * (10**-4)
    print("I_sppa: " + str(I_sppa) + " W/cm^2")

    # I_spta (720 mW/cm^2)
    tau = PD * 3000 # pulse repetition
    I_spta = I_sppa * tau
    I_spta = I_spta * 1000
    print("I_spta: " + str(I_spta) + " mW/cm^2")

    # TI
    A = (15 * 10 ** -3) * (25 * 10 ** -3) # area covered?
    W0 = (PII * 10 ** 3) * A # additional factor of (10 ** 3) for cm to m conversion
    TI = W0 / (210 / f)
    print("Thermal Index: " + str(TI) + "\n")

    return PII, MI, I_sppa, I_spta, TI

def derated(Pa):
    alpha = .3
    dist = 7
    f = 2.5
    dB  = alpha * dist * f
    der = np.exp(-dB/8.686)
    Pa_d = Pa * der
    return Pa_d


# main
if __name__ == '__main__':
    table = pd.read_excel('./P4-2_DopplerWaveform.xlsx')
    calibration = pd.read_table('./HGL-0085-SN1020_C20_20060405.txt')

    V = np.array(table[:][2])
    V = V[1:-1]
    t = np.array(table['x-axis'])
    t = t[1:-1]

    start = int(sum(np.where(t == .01464594)))
    stop = int(sum(np.where(t == .0146478)))

    # data preprocessing for this file specifically
    V_new = V[start - 11:stop - 1]
    t_new = t[start - 11:stop - 1]
    CV, Pa = conversionFactor(calibration,2.5,V_new)
    Pa = Pa - 1.2

    # medium specific data
    rho = 1000
    c = 1428
    z = rho * c

    FDAValues(Pa,t_new,2.5,z)

    Pa_d = derated(Pa)

    print("Derated values:")
    FDAValues(Pa_d, t_new, 2.5,z)
