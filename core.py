import scipy.io as scipyIO
import matplotlib.pyplot as plt
ECoG_Handpose = scipyIO.loadmat('./ECoG_Handpose')
ECoG_Handpose = ECoG_Handpose['y']
from sklearn import model_selection
import statistics as stats

# Data shape 67 x 507025

HYPER_PARAM = 31

import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn

import numpy as np
import random

# remove channels 0, 1, 5, 20, 23, 56

# Model Definition
class BetterCNN(nn.Module):
    
    def __init__(self):
        super(BetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(HYPER_PARAM, 128, (3, 3), padding=2)
        self.conv2 = nn.Conv2d(128, 64, (2, 2), padding=0)
        self.fc1 = nn.Linear(3072, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 3)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, (2,2))
        # out = self.conv2(out)
        # out = F.relu(out)
        # out = F.max_pool2d(out, (2,2))
        out = F.dropout(out, 0.2)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

# Model Definition
class OtherCNN(nn.Module):
    
    def __init__(self):
        super(OtherCNN, self).__init__()
        self.conv1 = nn.Conv2d(HYPER_PARAM, 128, (3, 3), padding=2)
        self.conv2 = nn.Conv2d(128, 64, (2, 2), padding=0)
        self.fc1 = nn.Linear(3072, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, (2,2))
        # out = self.conv2(out)
        # out = F.relu(out)
        # out = F.max_pool2d(out, (2,2))
        out = F.dropout(out, 0.2)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

# build the model and load state
model = BetterCNN()
model.train()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.BCEWithLogitsLoss()

from scipy.signal import butter, lfilter, sosfilt
from scipy.fft import fft, fftfreq
import scipy.signal as signal
fs = 1200.0  # Sample frequency (Hz)
T = 1/fs
f0 = 50.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor
N = ECoG_Handpose.shape[1]


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


highcut = 300
lowcut = 10

DATABOX = []
length = 0
for i in range(1,61):
    # if i not in [3, 4, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60]:
    y = ECoG_Handpose[i][1500:]
    length = len(y)
    x = plt.psd(y, len(y), 1200)
    plt.clf()
    window = signal.gaussian(len(x[0]), std=50000)
    plt.plot(window)
    plt.show()
    plt.clf()
    y = []
    for j in range(len(x[0])):
        y.append(x[0][j] * window[j])
    print(len(x[1]))
    plt.psd(y,512, 1200)
    plt.show()
    plt.clf()
    out = butter_bandpass_filter(y, lowcut, highcut, fs, order=6)
    DATABOX.append(out)
    if i ==10:
        exit()


# DATABOX = ECoG_Handpose[1:61][:]
DATABOX = np.array(DATABOX)
DATABOX = DATABOX.T
print(DATABOX.shape)

y_label = torch.zeros(1)

# y_labels = torch.from_numpy(ECoG_Handpose[61])
loss_log = []
acc_log = []
mylist = [ i for i in range(int( length / HYPER_PARAM )) ]
random.shuffle(mylist)
training_data, test_data = model_selection.train_test_split(mylist, test_size=0.2)


zero_guess = OtherCNN()
zero_guess.train()
zero_optimiser = torch.optim.Adam(zero_guess.parameters(), lr = 0.0001)


accuracy = 0
count = 0
acc_count = 1
zero_count = 0
z_acc = 0

for i in training_data:
    if count % 100 == 0:
        print(count/len(training_data))
    count += 1    
    if count % 1000 == 0:
        print('Running Accuracy')
        print(accuracy/acc_count)
        print('Running Zero Accuracy')
        print(z_acc/count)
    x_input = torch.zeros(HYPER_PARAM,6,10)
    optimiser.zero_grad()

    # Create the image
    for k in range(HYPER_PARAM):
        for j in range(6):
            x_input[k][j] = torch.from_numpy(DATABOX[i + k][j*10 : (j+1)*10])

    # Get current prediction
    x_input = x_input.unsqueeze(0)

    out = model(x_input)    
    tmp_out = out.detach().numpy()

    # if tmp_out[0][0] <= -0.1 and tmp_out[0][1] <= -0.1 and tmp_out[0][2] <= -0.1:
    #     arg = 0

    zero_out = zero_guess(x_input)

    y_label[0] = stats.mode(ECoG_Handpose[61][i*HYPER_PARAM:(i+1)*HYPER_PARAM])
    y_label = y_label.long()
    index = y_label.item()

    if index != 0:
        acc_count += 1
        log = torch.zeros(1,3)

        if index != 0:
            log[0][int(index) - 1] = 1

        # print(out)
        # print(y_label.item(), arg)
        # print(log)

        loss = criterion(out, log)

        if np.argmax(tmp_out) + 1 == index:
            accuracy += 1
        acc_log.append(accuracy/acc_count)

        loss_log.append(loss.item())

        model.train()
        loss.backward()
        optimiser.step()
        model.eval()

        if np.argmax(zero_out.detach().numpy()) == 1:
            z_acc += 1

    else:
        if np.argmax(zero_out.detach().numpy()) == 0:
            z_acc += 1

    log = torch.zeros(1,2)
    if index != 0:
        log[0][1] = 1
    else:
        log[0][0] = 1
        
    zero_guess.train()
    loss = criterion(zero_out, log)

    loss.backward()
    zero_optimiser.step()
    zero_guess.eval()


plt.title('Training Loss')
plt.plot(loss_log)
plt.show()
plt.clf()
plt.title('Training Accuracy')
plt.plot(acc_log)
plt.show()



loss_log = []
acc_log = []
accuracy = 0
count = 0
for i in test_data:
    count += 1    
    x_input = torch.zeros(HYPER_PARAM,6,10)
    # Create the image
    for k in range(HYPER_PARAM):
        for j in range(6):
            x_input[k][j] = torch.from_numpy(DATABOX[i + k][j*10 : (j+1)*10])
    # Get current prediction
    x_input = x_input.unsqueeze(0)
    out = model(x_input)    
    tmp_out = out.detach().numpy()
    zero_out = zero_guess(x_input)

    arg = 0
    if np.argmax(zero_out.detach().numpy()) == 1:
        arg = np.argmax(tmp_out) + 1

    y_label[0] = stats.mode(ECoG_Handpose[61][i*HYPER_PARAM:(i+1)*HYPER_PARAM])
    # log = torch.zeros(1,3)
    # index = y_label.item()

    # if index != 0:
    #     log[0][int(index) - 1] = 1.5

    print(out)
    # print(zero_out)
    print(y_label.item(), arg)


    # print(log)


    y_label = y_label.long()
    # loss = criterion(out, log)

    if arg == y_label.item():
        accuracy += 1

    acc_log.append(accuracy/count)
    # loss_log.append(loss.item())


# plt.title('Testing Loss')
# plt.plot(loss_log)
# plt.show()
plt.clf()
plt.title('Testing Accuracy')
plt.plot(acc_log)
plt.show()
print('TEST FINAL ACCURACY')
print(accuracy/count)