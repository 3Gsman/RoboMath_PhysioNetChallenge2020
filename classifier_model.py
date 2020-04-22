import numpy as np
from scipy.signal import butter, lfilter
from scipy import stats
import numpy as np
import os
import sys
from scipy.io import loadmat
from get_12ECG_features import detect_peaks
from sklearn import model_selection
from sklearn import tree
import joblib
from sklearn import preprocessing

# Function to load data from Driver.py
def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


# Funtion from get_123ECG_features, with a small change in the return, returning the labels
def get_12ECG_features(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            label = iline.split(': ')[1].split(',')[0]


    
#   We are only using data from lead1
    peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[0])
   
#   mean
    mean_RR = np.mean(idx/sample_Fs*1000)
    mean_Peaks = np.mean(peaks*gain_lead[0])

#   median
    median_RR = np.median(idx/sample_Fs*1000)
    median_Peaks = np.median(peaks*gain_lead[0])

#   standard deviation
    std_RR = np.std(idx/sample_Fs*1000)
    std_Peaks = np.std(peaks*gain_lead[0])

#   variance
    var_RR = stats.tvar(idx/sample_Fs*1000)
    var_Peaks = stats.tvar(peaks*gain_lead[0])

#   Skewness
    skew_RR = stats.skew(idx/sample_Fs*1000)
    skew_Peaks = stats.skew(peaks*gain_lead[0])

#   Kurtosis
    kurt_RR = stats.kurtosis(idx/sample_Fs*1000)
    kurt_Peaks = stats.kurtosis(peaks*gain_lead[0])

    features = np.hstack([age,sex,mean_RR,mean_Peaks,median_RR,median_Peaks,std_RR,std_Peaks,var_RR,var_Peaks,skew_RR,skew_Peaks,kurt_RR,kurt_Peaks])

  
    return features,label


# Training data input from command parameters
input_directory = sys.argv[1]

input_files = []

for f in os.listdir(input_directory):
  if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
    input_files.append(f)

data = []
header_data = []


print('Extracting 12ECG features...')
num_files = len(input_files)

for i, f in enumerate(input_files):
    print('    {}/{}...'.format(i+1, num_files))
    tmp_input_file = os.path.join(input_directory, f)
    data2, header_data2 = load_challenge_data(tmp_input_file)
    
    data.append(data2)
    header_data.append(header_data2)


file_data = []
for i in range(len(data)):
    file_data.append([data[i], header_data[i]])


features = []
labels = []

for item in file_data:
  feature,label = get_12ECG_features(item[0],item[1])
  features.append(feature)
  labels.append(label)

#Clean the labels
labels = [w.replace('\n', '') for w in labels]

#Encoding the labels
le = preprocessing.LabelEncoder()
labels2 = le.fit(labels)
labels2 = le.transform(labels)

X = features
Y = labels2

test_size = 0.5
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
print("Trainning model...")
model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

print("Model saved as",filename)

print("Script finished")