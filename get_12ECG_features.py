#!/usr/bin/env python

from scipy import stats
import os
import sys
from wfdb import processing
from scipy.signal import find_peaks
import pyhrv.nonlinear as nl
from contextlib import contextmanager
import pywt
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import numpy as np



@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/_modules/heartpy/filtering.html
'''
Functions for data filtering tasks.
'''

__all__ = ['filter_signal',
           'hampel_filter',
           'hampel_correcter',
           'smooth_signal']


def butter_lowpass(cutoff, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass', return_top=False):
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'highpass':
        b, a = butter_highpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        assert type(cutoff) == tuple or list or np.array, 'if bandpass filter is specified, \
cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff, Q=0.005, fs=sample_rate)
    else:
        raise ValueError('filtertype: %s is unknown, available are: \
lowpass, highpass, bandpass, and notch' % filtertype)

    filtered_data = filtfilt(b, a, data)

    if return_top:
        return np.clip(filtered_data, a_min=0, a_max=None)
    else:
        return filtered_data


def hurstFun(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses 
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def normalize_arr(data):
    norm = np.linalg.norm(data)
    normData = data / norm

    return normData


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def get_R_peaks(data, freq):
    with suppress_stdout():
        indexR = processing.xqrs_detect(sig=data, fs=freq)

    if len(indexR) > 2:

        indexR = indexR[1:]

        intervalR = np.diff(indexR)

        data = normalize_arr(data)

        valuesR = []
        for item in indexR:
            valuesR.append(data[item])

    else:

        data = reject_outliers(data)

        with suppress_stdout():
            indexR = processing.xqrs_detect(sig=data, fs=freq)

        if len(indexR) > 2:

            indexR = indexR[1:]

            intervalR = np.diff(indexR)

            data = normalize_arr(data)

            valuesR = []
            for item in indexR:
                valuesR.append(data[item])

        else:

            valuesR = [0]
            indexR = np.array([0])
            intervalR = np.array([0])

    return valuesR, indexR, intervalR


def find_nearest_previous(array, value):
    idx = np.searchsorted(array, value, side="left")
    return idx, array[idx - 1]


def get_Q_peaks(data, freq, indexR):
    indexQtemp = processing.gqrs_detect(sig=data, fs=freq)

    if len(indexQtemp) > 2 and len(indexR) > 2:

        indexQ = np.array([])

        for item in indexR:
            indexQ = np.append(indexQ, find_nearest_previous(indexQtemp, item)[1])

        indexQ = indexQ.astype(int)

        # indexQ = indexQ[1:]

        intervalQ = np.diff(indexQ)

        data = normalize_arr(data)

        valuesQ = []
        for item in indexQ:
            valuesQ.append(data[item])

    else:

        valuesQ = [0]
        indexQ = np.array([0])
        intervalQ = np.array([0])

    return valuesQ, indexQ, intervalQ


def s_t_peaks(data, indexR):
    if len(indexR) > 2:

        data = normalize_arr(data)

        dataInput = data

        qrs_inds = indexR

        x = data
        peaks, _ = find_peaks(x, distance=50)

        rr = qrs_inds
        peaks = peaks.tolist()
        rr2 = rr.tolist()

        peaksArr = []
        peaksVal = []

        for i in range(len(rr) - 1):
            tempArr = []
            tempArr2 = []

            for f in peaks:
                if f > rr2[i] + 50 and f < rr2[i + 1] - 50:
                    tempArr.append(f)
                    tempArr2.append(x[f])

            peaksArr.append(tempArr)
            peaksVal.append(tempArr2)

        tIndex = []
        tVal = []

        for i in range(len(peaksVal)):
            qs = dict(zip(peaksArr[i], peaksVal[i]))
            qsFound = sorted(qs, key=qs.get)
            # qsArr.append(qsFound[-2:])
            if len(qsFound) > 0:
                indexTop = qsFound[-1]
                tIndex.append(indexTop)
                tVal.append(x[indexTop])
            else:
                tIndex.append(0)
                tVal.append(0)

        x = -data
        valleys, _ = find_peaks(x)

        valleys = valleys.tolist()

        valleysArr = []
        valleysVal = []

        for i in range(len(rr) - 1):
            tempArr = []
            tempArr2 = []

            for f in valleys:
                if f > rr2[i] and f < tIndex[i]:
                    tempArr.append(f)
                    tempArr2.append(x[f])

            valleysArr.append(tempArr)
            valleysVal.append(tempArr2)

        sIndex = []
        sVal = []

        for i in range(len(valleysVal)):
            qs = dict(zip(valleysArr[i], valleysVal[i]))
            qsFound = sorted(qs, key=qs.get)
            # qsArr.append(qsFound[-2:])
            if len(qsFound) > 0:
                indexTop = qsFound[-1]
                sIndex.append(indexTop)
                sVal.append(x[indexTop])
            else:
                sIndex.append(0)
                sVal.append(0)

        intervalT = np.diff(tIndex)
        intervalS = np.diff(sIndex)

    else:

        tIndex = [0]
        tVal = [0]
        intervalT = np.array([0])

        sIndex = [0]
        sVal = [0]
        intervalS = np.array([0])

    return tIndex, tVal, intervalT, sIndex, sVal, intervalS


def get_peaks_distances(indexQ, indexR, indexS, indexT, data):
    if len(indexR) > 2:

        disQR = []
        disQS = []
        disQT = []
        disRS = []
        disRT = []
        disST = []

        disQRVal = []
        disQSVal = []
        disQTVal = []
        disRSVal = []
        disRTVal = []
        disSTVal = []

        for i in range(len(indexR) - 1):
            disQR.append(indexR[i] - indexQ[i])
            disQRVal.append(data[indexQ[i]:indexR[i]])

            disQS.append(indexS[i] - indexQ[i])
            disQSVal.append(data[indexQ[i]:indexS[i]])

            disQT.append(indexT[i] - indexQ[i])
            disQTVal.append(data[indexQ[i]:indexT[i]])

            disRS.append(indexS[i] - indexR[i])
            disRSVal.append(data[indexR[i]:indexS[i]])

            disRT.append(indexT[i] - indexR[i])
            disRTVal.append(data[indexR[i]:indexT[i]])

            disST.append(indexT[i] - indexS[i])
            disSTVal.append(data[indexS[i]:indexT[i]])

    else:

        disQR = [0]
        disQS = [0]
        disQT = [0]
        disRS = [0]
        disRT = [0]
        disST = [0]

        disQRVal = [[0]]
        disQSVal = [[0]]
        disQTVal = [[0]]
        disRSVal = [[0]]
        disRTVal = [[0]]
        disSTVal = [[0]]

    return [disQR, disQS, disQT, disRS, disRT, disST], [disQRVal, disQSVal, disQTVal, disRSVal, disRTVal, disSTVal]


def peaks_derivations(indexPeaks, derivation):
    derValues = []

    for item in indexPeaks:
        derValues.append(derivation[item])

    return (derValues)


def wav_info(data):
    coef, freqs = pywt.cwt(data, [1, 30, 60], 'gaus1')

    return coef[0], coef[1], coef[2]


# Funtion from get_123ECG_features, with a small change in the return, returning the labels
def get_12ECG_featuresTrain(data, header_data):
    printInfo = 0
    poincareActivation = 1
    
    if printInfo == 1:
        print("header_data")
        print(header_data)
        print("data")
        print(data)
    
    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs = int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii + 1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])
    
    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip() == 'Female':
                sex = 1
            else:
                sex = 0
        elif iline.startswith('#Dx'):
            label = [iline.split(': ')[1]]
    
    filteredData = filter_signal(data[0], 40, 500, order=2, filtertype='lowpass', return_top=False)
    
    valuesR, indexR, intervalR = get_R_peaks(filteredData, sample_Fs)
    
    valuesQ, indexQ, intervalQ = get_Q_peaks(filteredData, sample_Fs, indexR)
    
    indexT, valuesT, intervalT, indexS, valuesS, intervalS = s_t_peaks(filteredData, indexR)
    
    dis = get_peaks_distances(indexQ, indexR, indexS, indexT, data[0])
    
    wav1, wav30, wav60 = wav_info(data[0])
    
    #if tsfelActivation == 1:
        #cfg = tsfel.get_features_by_domain()
    
        #tsfelFeatList = []
        # Extract features
        #with suppress_stdout():
            #tsfelFeat = tsfel.time_series_features_extractor(cfg, filteredData, fs=sample_Fs)
        #tsfelFeatList = tsfelFeat.values.tolist()[0]
    
    ecgVals = []
    derivationsValues = []
    
    for i in range(12):
    
        lead = normalize_arr(data[i])
    
        ecgVals.append(np.mean(lead))
        ecgVals.append(np.median(lead))
        ecgVals.append(np.std(lead))
        ecgVals.append(stats.tvar(lead))
        ecgVals.append(stats.skew(lead))
        ecgVals.append(stats.kurtosis(lead))
        ecgVals.append(np.amax(lead))
        ecgVals.append(np.amin(lead))
        ecgVals.append(hurstFun(lead))
    
        if i > 0:
            derivationsValues.append(peaks_derivations(indexQ, lead))
            derivationsValues.append(peaks_derivations(indexR, lead))
            derivationsValues.append(peaks_derivations(indexS, lead))
            derivationsValues.append(peaks_derivations(indexT, lead))
    
    timeSeries = [valuesR, intervalR, valuesQ, intervalQ, valuesT, intervalT, valuesS, intervalS, wav1, wav30, wav60]
    timeSeries = timeSeries + derivationsValues
    
    peaksStats = []
    
    for item in timeSeries:
        peaksStats.append(np.amax(item))
        peaksStats.append(np.amin(item))
        peaksStats.append(np.mean(item))
        peaksStats.append(np.median(item))
        peaksStats.append(np.std(item))
        peaksStats.append(stats.tvar(item))
        peaksStats.append(stats.skew(item))
        peaksStats.append(stats.kurtosis(item))
        peaksStats.append(hurstFun(item))
    
    distancesPeaks = []
    
    for item in dis[0]:
        distancesPeaks.append(np.mean(item))
        distancesPeaks.append(np.median(item))
        distancesPeaks.append(np.std(item))
        distancesPeaks.append(stats.tvar(item))
        distancesPeaks.append(stats.skew(item))
        distancesPeaks.append(stats.kurtosis(item))
        distancesPeaks.append(np.amax(item))
        distancesPeaks.append(np.amin(item))
        distancesPeaks.append(hurstFun(item))
    
    distancesPeaksVals = []
    
    for item in dis[1]:
    
        distancesPeaksValsInd = [[], [], []]
    
        for segment in item:
    
            if len(segment) > 0:
    
                linReg = stats.linregress(np.arange(len(segment)), segment)
    
                distancesPeaksValsInd[0].append(np.mean(segment))
                distancesPeaksValsInd[1].append(np.std(segment))
                distancesPeaksValsInd[2].append(linReg[0])
    
    
            else:
    
                distancesPeaksValsInd[0].append(0)
                distancesPeaksValsInd[1].append(0)
                distancesPeaksValsInd[2].append(0)
    
        for sumary in distancesPeaksValsInd:
            distancesPeaksVals.append(np.mean(sumary))
            distancesPeaksVals.append(np.median(sumary))
            distancesPeaksVals.append(np.std(sumary))
            distancesPeaksVals.append(stats.tvar(sumary))
            distancesPeaksVals.append(stats.skew(sumary))
            distancesPeaksVals.append(stats.kurtosis(sumary))
            distancesPeaksVals.append(np.amax(sumary))
            distancesPeaksVals.append(np.amin(sumary))
    
    if poincareActivation == 1:
        pcData = []
        poincare = nl.poincare(intervalR, show=False)
        pcData.append(poincare[1])
        pcData.append(poincare[2])
        pcData.append(poincare[3])
        pcData.append(poincare[4])
    
    if printInfo == 1:
        print("valuesQ")
        print(valuesQ)
        print("indexQ")
        print(indexQ)
        print("intervalQ")
        print(intervalQ)
    
        print("valuesR")
        print(valuesR)
        print("indexR")
        print(indexR)
        print("intervalR")
        print(intervalR)
    
        print("valuesS")
        print(valuesS)
        print("indexS")
        print(indexS)
        print("intervalS")
        print(intervalS)
    
        print("valuesT")
        print(valuesT)
        print("indexT")
        print(indexT)
        print("intervalT")
        print(intervalT)
    
        print("distancesPeaks")
        print(distancesPeaks)
        print("distancesPeaks[3]")
        print(distancesPeaks[3])
    
        # if tsfelActivation == 1:
        # print("tsfelFeatList")
        # print(tsfelFeatList)
    
    #   FEAT to list
    
    features = np.hstack([age, sex])
    
    features = np.concatenate((features, ecgVals))
    
    if poincareActivation == 1:
        features = np.concatenate((features, pcData))
    
    features = np.concatenate((features, peaksStats))
    features = np.concatenate((features, distancesPeaks))
    features = np.concatenate((features, distancesPeaksVals))
    
    #if tsfelActivation == 1:
        #features = np.concatenate((features, tsfelFeatList))
    
    return features, label


def get_12ECG_features(data, header_data):
    printInfo = 0
    poincareActivation = 1
    #tsfelActivation = 0

    if printInfo == 1:
        print("header_data")
        print(header_data)
        print("data")
        print(data)

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs = int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)

    for ii in range(num_leads):
        tmp_hea = header_data[ii + 1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip() == 'Female':
                sex = 1
            else:
                sex = 0
        elif iline.startswith('#Dx'):
            label = [iline.split(': ')[1]]

    filteredData = filter_signal(data[0], 40, 500, order=2, filtertype='lowpass', return_top=False)

    valuesR, indexR, intervalR = get_R_peaks(filteredData, sample_Fs)

    valuesQ, indexQ, intervalQ = get_Q_peaks(filteredData, sample_Fs, indexR)

    indexT, valuesT, intervalT, indexS, valuesS, intervalS = s_t_peaks(filteredData, indexR)

    dis = get_peaks_distances(indexQ, indexR, indexS, indexT, data[0])

    wav1, wav30, wav60 = wav_info(data[0])

    #if tsfelActivation == 1:
        #cfg = tsfel.get_features_by_domain()

        #tsfelFeatList = []
        # Extract features
        #with suppress_stdout():
            #tsfelFeat = tsfel.time_series_features_extractor(cfg, filteredData, fs=sample_Fs)
        #tsfelFeatList = tsfelFeat.values.tolist()[0]

    #   ECG vals

    ecgVals = []
    derivationsValues = []

    for i in range(12):

        lead = normalize_arr(data[i])

        ecgVals.append(np.mean(lead))
        ecgVals.append(np.median(lead))
        ecgVals.append(np.std(lead))
        ecgVals.append(stats.tvar(lead))
        ecgVals.append(stats.skew(lead))
        ecgVals.append(stats.kurtosis(lead))
        ecgVals.append(np.amax(lead))
        ecgVals.append(np.amin(lead))
        ecgVals.append(hurstFun(lead))

        if i > 0:
            derivationsValues.append(peaks_derivations(indexQ, lead))
            derivationsValues.append(peaks_derivations(indexR, lead))
            derivationsValues.append(peaks_derivations(indexS, lead))
            derivationsValues.append(peaks_derivations(indexT, lead))

    timeSeries = [valuesR, intervalR, valuesQ, intervalQ, valuesT, intervalT, valuesS, intervalS, wav1, wav30, wav60]
    timeSeries = timeSeries + derivationsValues

    peaksStats = []

    for item in timeSeries:
        peaksStats.append(np.amax(item))
        peaksStats.append(np.amin(item))
        peaksStats.append(np.mean(item))
        peaksStats.append(np.median(item))
        peaksStats.append(np.std(item))
        peaksStats.append(stats.tvar(item))
        peaksStats.append(stats.skew(item))
        peaksStats.append(stats.kurtosis(item))
        peaksStats.append(hurstFun(item))

    distancesPeaks = []

    for item in dis[0]:
        distancesPeaks.append(np.mean(item))
        distancesPeaks.append(np.median(item))
        distancesPeaks.append(np.std(item))
        distancesPeaks.append(stats.tvar(item))
        distancesPeaks.append(stats.skew(item))
        distancesPeaks.append(stats.kurtosis(item))
        distancesPeaks.append(np.amax(item))
        distancesPeaks.append(np.amin(item))
        distancesPeaks.append(hurstFun(item))

    distancesPeaksVals = []

    for item in dis[1]:

        distancesPeaksValsInd = [[], [], []]

        for segment in item:

            if len(segment) > 0:

                linReg = stats.linregress(np.arange(len(segment)), segment)

                distancesPeaksValsInd[0].append(np.mean(segment))
                distancesPeaksValsInd[1].append(np.std(segment))
                distancesPeaksValsInd[2].append(linReg[0])


            else:

                distancesPeaksValsInd[0].append(0)
                distancesPeaksValsInd[1].append(0)
                distancesPeaksValsInd[2].append(0)

        for sumary in distancesPeaksValsInd:
            distancesPeaksVals.append(np.mean(sumary))
            distancesPeaksVals.append(np.median(sumary))
            distancesPeaksVals.append(np.std(sumary))
            distancesPeaksVals.append(stats.tvar(sumary))
            distancesPeaksVals.append(stats.skew(sumary))
            distancesPeaksVals.append(stats.kurtosis(sumary))
            distancesPeaksVals.append(np.amax(sumary))
            distancesPeaksVals.append(np.amin(sumary))

    if poincareActivation == 1:
        pcData = []
        poincare = nl.poincare(intervalR, show=False)
        pcData.append(poincare[1])
        pcData.append(poincare[2])
        pcData.append(poincare[3])
        pcData.append(poincare[4])

    if printInfo == 1:
        print("valuesQ")
        print(valuesQ)
        print("indexQ")
        print(indexQ)
        print("intervalQ")
        print(intervalQ)

        print("valuesR")
        print(valuesR)
        print("indexR")
        print(indexR)
        print("intervalR")
        print(intervalR)

        print("valuesS")
        print(valuesS)
        print("indexS")
        print(indexS)
        print("intervalS")
        print(intervalS)

        print("valuesT")
        print(valuesT)
        print("indexT")
        print(indexT)
        print("intervalT")
        print(intervalT)

        print("distancesPeaks")
        print(distancesPeaks)
        print("distancesPeaks[3]")
        print(distancesPeaks[3])

        # if tsfelActivation == 1:
        # print("tsfelFeatList")
        # print(tsfelFeatList)

    #   FEAT to list

    features = np.hstack([age, sex])

    features = np.concatenate((features, ecgVals))

    if poincareActivation == 1:
        features = np.concatenate((features, pcData))

    features = np.concatenate((features, peaksStats))
    features = np.concatenate((features, distancesPeaks))
    features = np.concatenate((features, distancesPeaksVals))

    #if tsfelActivation == 1:
        #features = np.concatenate((features, tsfelFeatList))

    return features