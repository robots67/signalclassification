import csv
from scipy import signal, fftpack
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

np.set_printoptions(suppress=True, threshold = np.inf)

t = np.linspace(0, 1, 1000, endpoint=False)
#Base signals
sig_sin = np.sin(2 * np.pi * 5 * t)
sig_square = signal.square(2 * np.pi * 5 * t)
sig_sawtooth = signal.sawtooth(2 * np.pi * 5 * t)

labels = ["signal_type"]

for i in range(1000):
    labels = labels + ["point_%i"%i]

def genBase():
    #Generate a dataset of just the three base signals
    with open('baseset.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        
        #label signals
        sin = np.insert(sig_sin, 0, 0., axis=0)
        square = np.insert(sig_square, 0, 1., axis=0)
        saw = np.insert(sig_sawtooth, 0, 2., axis=0)
        #write rows
        sigwriter.writerow(sin)
        sigwriter.writerow(square)
        sigwriter.writerow(saw)
    print("Generated baseset!")

def genNoise():
    #Generate a dataset of signals with noise then multiply amplitude randomly
    with open('noisetrain.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        for row in range(500):        
            sin_noise = ((sig_sin + np.random.randn(len(sig_sin)))
                    *np.random.randint(1,10))
            sin_noise = np.insert(sin_noise, 0, 0., axis=0)
            sigwriter.writerow(sin_noise)
            square_noise = ((sig_square + np.random.randn(len(sig_square)))
                    *np.random.randint(1,10))
            square_noise = np.insert(square_noise, 0, 1., axis=0)
            sigwriter.writerow(square_noise)
            saw_noise = ((sig_sawtooth + np.random.randn(len(sig_sawtooth)))
                    *np.random.randint(1,10))
            saw_noise = np.insert(saw_noise, 0, 2., axis=0)
            sigwriter.writerow(saw_noise)
    with open('noisetest.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        for row in range(500):        
            sin_noise = ((sig_sin + np.random.randn(len(sig_sin)))
                *np.random.randint(1,10))
            sin_noise = np.insert(sin_noise, 0, 0., axis=0)
            sigwriter.writerow(sin_noise)
        for row in range(500):
            square_noise = ((sig_square + np.random.randn(len(sig_square)))
                *np.random.randint(1,10))
            square_noise = np.insert(square_noise, 0, 1., axis=0)
            sigwriter.writerow(square_noise)
        for row in range(500):
            saw_noise = ((sig_sawtooth + np.random.randn(len(sig_sawtooth)))
                *np.random.randint(1,10))
            saw_noise = np.insert(saw_noise, 0, 2., axis=0)
            sigwriter.writerow(saw_noise)
    print("Generated noiseset!")

def genTest():
    #Generate test dataset with random noise amplitude multiplier between 1-10
    with open('signaltest.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        for row in range(1000):
            sin_noise = (sig_sin + np.random.randn(len(sig_sin))
                *np.random.randint(1,10))
            sin_noise = np.insert(sin_noise, 0, 0., axis=0)
            sigwriter.writerow(sin_noise)
        for row in range(1000):
            square_noise = (sig_square + np.random.randn(len(sig_square))
                *np.random.randint(1,10))
            square_noise = np.insert(square_noise, 0, 1., axis=0)
            sigwriter.writerow(square_noise)
        for row in range(1000):
            saw_noise = (sig_sawtooth + np.random.randn(len(sig_sawtooth))
                *np.random.randint(1,10))
            saw_noise = np.insert(saw_noise, 0, 2., axis=0)
            sigwriter.writerow(saw_noise)
    with open('signaltrain.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        for row in range(1000):
            sin_noise = (sig_sin + np.random.randn(len(sig_sin))
                *np.random.randint(1,10))
            sin_noise = np.insert(sin_noise, 0, 0., axis=0)
            sigwriter.writerow(sin_noise)
            square_noise = (sig_square + np.random.randn(len(sig_square))
                *np.random.randint(1,10))
            square_noise = np.insert(square_noise, 0, 1., axis=0)
            sigwriter.writerow(square_noise)
            saw_noise = (sig_sawtooth + np.random.randn(len(sig_sawtooth))
                *np.random.randint(1,10))
            saw_noise = np.insert(saw_noise, 0, 2., axis=0)
            sigwriter.writerow(saw_noise)
    print("Generated signaltest!")

def genPartial():
    #Generate a dataset of partial (shorter) signals
    part_t = np.linspace(0, 1, 300, endpoint=False)
    #Base signals
    part_sin = np.sin(2 * np.pi * 5 * part_t)
    part_square = signal.square(2 * np.pi * 5 * part_t)
    part_saw = signal.sawtooth(2 * np.pi * 5 * part_t)

    with open('partialset.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels[0:301])
        for row in range(1000):
            sin_noise = (sig_sin + np.random.randn(len(part_sin))
                *np.random.randint(1,10))
            sin_noise = np.insert(sin_noise, 0, 0., axis=0)
            sigwriter.writerow(sin_noise)
        for row in range(1000):
            square_noise = (sig_square + np.random.randn(len(part_square))
                *np.random.randint(1,10))
            square_noise = np.insert(square_noise, 0, 1., axis=0)
            sigwriter.writerow(square_noise)
        for row in range(1000):
            saw_noise = (sig_sawtooth + np.random.randn(len(part_saw))
                *np.random.randint(1,10))
            saw_noise = np.insert(saw_noise, 0, 2., axis=0)
            sigwriter.writerow(saw_noise)
    print("Generated partialsets!")

def genCorrSets():
    #Generate correlation training and testing sets
    #Adds noise of random amplitude
    def corr(sig):
        #correlate the signal
        corr_ones = signal.correlate(sig, np.ones(64), mode='same')/64
        return corr_ones

    with open('correlationtrain.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        for row in range(500):
            sin_noise = (sig_sin + np.random.randn(len(sig_sin))
                *np.random.randint(1,10))
            corr_sig = corr(sin_noise)
            corr_sig = np.insert(corr_sig, 0, 0., axis=0)
            sigwriter.writerow(corr_sig)
        for row in range(500):
            square_noise = (sig_square + np.random.randn(len(sig_square))
                *np.random.randint(1,10))
            corr_sig = corr(square_noise)
            corr_sig = np.insert(corr_sig, 0, 1., axis=0)
            sigwriter.writerow(corr_sig)
        for row in range(500):
            saw_noise = (sig_sawtooth + np.random.randn(len(sig_sawtooth))
                *np.random.randint(1,10))
            corr_sig = corr(saw_noise)
            corr_sig = np.insert(corr_sig, 0, 2., axis=0)
            sigwriter.writerow(corr_sig)
    with open('correlationtest.csv', 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        for row in range(500):
            sin_noise = (sig_sin + np.random.randn(len(sig_sin))
                *np.random.randint(1,10))
            corr_sig = corr(sin_noise)
            corr_sig = np.insert(corr_sig, 0, 0., axis=0)
            sigwriter.writerow(corr_sig)
        for row in range(500):
            square_noise = (sig_square + np.random.randn(len(sig_square))
                *np.random.randint(1,10))
            corr_sig = corr(square_noise)
            corr_sig = np.insert(corr_sig, 0, 1., axis=0)
            sigwriter.writerow(corr_sig)
        for row in range(500):
            saw_noise = (sig_sawtooth + np.random.randn(len(sig_sawtooth))
                *np.random.randint(1,10))
            corr_sig = corr(saw_noise)
            corr_sig = np.insert(corr_sig, 0, 2., axis=0)
            sigwriter.writerow(corr_sig)
    print("Generated corrsets!")

def genFilteredSet():
    #Generates a set that is filtered through a LOWESS function to smooth noise
    #More effective than cross correlation but also significantly slower
    with open("filtertrain.csv", 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile,delimiter= ',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        points = t * 1000
        for row in range(1500):
            sin_noise = (sig_sin + np.random.randn(len(sig_sin)) 
                * np.random.randint(1,10))
            filter_sin =lowess(sin_noise, points,
                is_sorted=True, frac=0.045, it=0)
            filter_sin = np.insert(filter_sin[:,1],0,0.,axis=0)
            sigwriter.writerow(filter_sin)
            square_noise = (sig_square + np.random.randn(len(sig_square)) 
                * np.random.randint(1,10))
            filter_square =lowess(square_noise, points,
                is_sorted=True, frac=0.045, it=0)
            filter_square = np.insert(filter_square[:,1],0,1.,axis=0)
            sigwriter.writerow(filter_square)
            saw_noise = (sig_sawtooth + np.random.randn(len(sig_sawtooth)) 
                * np.random.randint(1,10))
            filter_saw =lowess(saw_noise, points,
                is_sorted=True, frac=0.045, it=0)
            filter_saw = np.insert(filter_saw[:,1],0,2.,axis=0)
            sigwriter.writerow(filter_saw)
    with open("filtertest.csv", 'w', newline='') as csvfile:
        sigwriter = csv.writer(csvfile,delimiter= ',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        sigwriter.writerow(labels)
        for row in range(500):
            sin_noise = (sig_sin + np.random.randn(len(sig_sin)) 
                * np.random.randint(1,10))
            filter_sin =lowess(sin_noise, points,
                is_sorted=True, frac=0.045, it=0)
            filter_sin = np.insert(filter_sin[:,1],0,0.,axis=0)
            sigwriter.writerow(filter_sin)
        for row in range(500):
            square_noise = (sig_square + np.random.randn(len(sig_square)) 
                * np.random.randint(1,10))
            filter_square =lowess(square_noise, points,
                is_sorted=True, frac=0.045, it=0)
            filter_square = np.insert(filter_square[:,1],0,1.,axis=0)
            sigwriter.writerow(filter_square)
        for row in range(500):
            saw_noise = (sig_sawtooth + np.random.randn(len(sig_sawtooth)) 
                * np.random.randint(1,10))
            filter_saw =lowess(saw_noise, points,
                is_sorted=True, frac=0.045, it=0)
            filter_saw = np.insert(filter_saw[:,1],0,2.,axis=0)
            sigwriter.writerow(filter_saw)
    print("Generated filtersets!")
