import scipy.io as scipyIO
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, lfilter
import random

#%%
ECoG_Handpose = scipyIO.loadmat('./ECoG_Handpose')
ECoG_Handpose = ECoG_Handpose['y']

ECoGchannels = ECoG_Handpose[1:61, 18:]
ECoGchannels = np.array(ECoGchannels)
stimuli = ECoG_Handpose[61, 18:]

#%%
# Butterworth filter applied (50-300 Hz)
data = []
highcut = 300
lowcut = 50
freq = 1200

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

for i in range(ECoGchannels.shape[0]):
    y = ECoGchannels[i]
    out = butter_bandpass_filter(y, lowcut, highcut, freq, order=6)
    data.append(out)
    
data = np.array(data)

#%%
# find indexs where the trial changes
changes_stim = []
for idx in range(1,len(stimuli)):
    if stimuli[idx-1] != stimuli[idx]:
        changes_stim.append(idx)
        
#%%
# make matrix of 67 components
pca = PCA(n_components=60)
pca.fit(data.T[1000:,:])
explained_var = pca.explained_variance_ratio_
plt.plot(np.cumsum(explained_var))
# observe the first 30 components are sort of okay

#%%
# transform the dataset
reduced_data = PCA(n_components=30).fit_transform(data.T[1000:,:])

print(reduced_data.shape)
input_feature_matrix = []
labels = []
for i in range(len(reduced_data)):
    input_feature_matrix.append(reduced_data[i])
    labels.append(ECoG_Handpose[61][i+1000])



kmeans = KMeans(n_clusters=6, random_state=0).fit(input_feature_matrix)
pred_labels = kmeans.labels_

zero_log = []
one_log = []
two_log = []
three_log = []
for l in range(len(pred_labels)):
    if labels[l] == 0:
        zero_log.append(pred_labels[l]+1)
    elif labels[l] == 1:
        one_log.append(pred_labels[l]+1)
    elif labels[l] == 2:
        two_log.append(pred_labels[l]+1)
    elif labels[l] == 3:
        three_log.append(pred_labels[l]+1)
    
print(np.mean(zero_log))
print(np.mean(one_log))
print(np.mean(two_log))
print(np.mean(three_log))
