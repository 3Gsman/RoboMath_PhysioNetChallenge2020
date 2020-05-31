from sklearn import preprocessing
import os
import sys
from scipy.io import loadmat
from sklearn import neural_network
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas
import numpy as np

from get_12ECG_features import get_12ECG_featuresTrain



# Function to load data from Driver.py
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


def classifier_model():
    print("________________________________")
    print('Script started')

    # Training data input from command parameters
    input_directory = sys.argv[1]

    input_files = []

    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)

    data = []
    header_data = []

    num_files = len(input_files)

    print("________________________________")
    print("Loading files...")

    for i, f in enumerate(input_files):
        print('File loaded {}/{}'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data2, header_data2 = load_challenge_data(tmp_input_file)

        data.append(data2)
        header_data.append(header_data2)

    file_data = []
    for i in range(len(data)):
        file_data.append([data[i], header_data[i]])

    features = []
    labels = []

    numb = 0
    numbErr = 0
    errorList = []

    print("________________________________")
    print("Starting features extraction...")

    for item in file_data:

        numb += 1

        try:
            feature, label = get_12ECG_featuresTrain(item[0], item[1])
            features.append(feature)
            labels.append(label)

        except:
            numbErr += 1
            errFile = item[1][0][:5]
            errorList.append(errFile)
            print("ERR:", errFile)

        print("Extracting features", numb, "/", len(file_data))

    print("Featured extraction done")
    print("Total Errors:", numbErr)
    if (numbErr > 0):
        print("Errors files:", errorList)

    # Clean the labels
    labels = [w[0].replace('\n', '') for w in labels]
    labels = [i.split(',') for i in labels]

    binaryLabels = MultiLabelBinarizer().fit_transform(labels)

    features = np.asarray(features)
    features = np.nan_to_num(features)
    features = features.tolist()

    data = features
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(data)
    cleanFeatures = scaler.transform(data)

    # Encoding the labels
    le = preprocessing.LabelEncoder()
    labels2 = le.fit(labels[0])
    labels2 = le.transform(labels[0])

    df1 = pandas.DataFrame(cleanFeatures)
    df2 = pandas.DataFrame(binaryLabels)

    df3 = pandas.concat([df1, df2], axis=1)

    df3.to_csv("checkpoint.csv")

    # print(labels2)

    X = cleanFeatures
    Y = binaryLabels

    # test_size = 0.1
    # seed = 7
    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Fit the model on training set
    print("________________________________")
    print("Trainning model...")
    model = MultiOutputClassifier(neural_network.MLPClassifier())
    model.fit(X, Y)
    print("________________________________")
    print("Model trained")
    # save the model to disk
    filename = 'finalized_model.sav'
    filenameScaler = 'std_scaler.bin'
    joblib.dump(model, filename)
    joblib.dump(scaler, filenameScaler)

    print("Model saved as", filename)
    print("Scaler saved as", filenameScaler)

    # from sklearn.metrics import classification_report,confusion_matrix
    # predictions=model.predict(X_test)

    # print(classification_report(Y_test,predictions))

    print("________________________________")
    print("Script finished")


classifier_model()