import numpy as np
from nptdms import TdmsFile
import pandas as pd
from FeatureExtraction import extract_features, perform_PCA_reduction
from neural_network_maker import make_model_ANN, make_model_CNN
from network_visualiser import make_plots_history, export_data_to_excel
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

random_state = 2500

# Dane Chirp

tdms_file1 = TdmsFile.read('DataFileKotwalipiec23_pociete_Gora_Dol_Zalom.tdms')

kotwa_ids = []
rodzaj_id = []
wzbudzenia = []
podjazd =[]

for group in tdms_file1.groups():
    for channel in group.channels():
        data_len = np.shape(channel.data)[0]
        if(data_len<=1000000 and data_len>=10000 and channel.name[0]!='Z'):
        #if(data_len<=1000000 and data_len>=10000):
            kotwa_ids.append(group.name)
            rodzaj_id.append(channel.name[0])
            podjazd.append(channel.name[2:])
            wzbudzenia.append(channel.data)

print(rodzaj_id)
#przyjęte że gora i zalom to ten sam rodzaj oznaczone 0, dol oznaczone 1, zalom odrzucony wyżej
rodzaj_klasyfikacja=[1 if elem == 'D' else 0 for elem in rodzaj_id]
print(rodzaj_klasyfikacja)

# plt.figure()
# plt.hist(rodzaj_klasyfikacja, 2)
# plt.title("Dane chirp podział")
# plt.show()

print(np.shape(rodzaj_klasyfikacja))

values, counts = np.unique(rodzaj_klasyfikacja, return_counts=True)
data_len = np.shape(rodzaj_klasyfikacja)[0]
print(counts)

# Dane SAS

tdms_file2 = TdmsFile.read('daneSASpociete.tdms')
tdms_file3 = TdmsFile.read('daneSASpociete2.tdms')
tdms_file4 = TdmsFile.read('daneSASkwiecienPociete.tdms')

kotwa_ids_sas = []
rodzaj_id_sas = []
wzbudzenia_sas = []
podjazd_sas =[]

for group in tdms_file2.groups():
    for channel in group.channels():
        data_len = np.shape(channel.data)[0]
        #if(data_len<=1000000 and data_len>=10000):
        if(data_len<=1000000 and data_len>=10000 and channel.name[0]!='Z'):
            kotwa_ids_sas.append(group.name)
            rodzaj_id_sas.append(channel.name[0])
            podjazd_sas.append(channel.name[2:])
            wzbudzenia_sas.append(channel.data)
            #print(channel.name[2:])
            #print(np.shape(channel.data))

for group in tdms_file3.groups():
    for channel in group.channels():
        data_len = np.shape(channel.data)[0]
        #if(data_len<=1000000 and data_len>=10000):
        if(data_len<=1000000 and data_len>=10000 and channel.name[0]!='Z'):
            kotwa_ids_sas.append(group.name)
            rodzaj_id_sas.append(channel.name[0])
            podjazd_sas.append(channel.name[2:])
            wzbudzenia_sas.append(channel.data)
            #print(channel.name[2:])
            #print(np.shape(channel.data))

for group in tdms_file4.groups():
    for channel in group.channels():
        data_len = np.shape(channel.data)[0]
        #if(data_len<=1000000 and data_len>=10000):
        if(data_len<=1000000 and data_len>=10000 and channel.name[0]!='Z'):
            kotwa_ids_sas.append(group.name)
            rodzaj_id_sas.append(channel.name[0])
            podjazd_sas.append(channel.name[2:])
            wzbudzenia_sas.append(channel.data)
            #print(channel.name[2:])
            #print(np.shape(channel.data))

# print(rodzaj_id_sas)
#przyjęte że gora i zalom to ten sam rodzaj oznaczone 0, dol oznaczone 1, zalom odrzucony wyżej
rodzaj_klasyfikacja_sas=[1 if elem == 'D' else 0 for elem in rodzaj_id_sas]
# print(rodzaj_klasyfikacja_sas)
# plt.figure()
# plt.hist(rodzaj_klasyfikacja_sas, 2)
# plt.title("Dane SAS podział")
# plt.show()

# print(np.shape(rodzaj_klasyfikacja_sas))

values_sas, counts_sas = np.unique(rodzaj_klasyfikacja_sas, return_counts=True)
data_len_sas = np.shape(rodzaj_klasyfikacja_sas)[0]
print(counts_sas)

#Przygotwanie wsadu
widma, stft_amplitudes, mfccs_all, wavelet_trans_all = extract_features(wzbudzenia, "chirp", make_plots=False)
widma_sas, stft_amplitudes_sas, mfccs_all_sas, wavelet_trans_all_sas = extract_features(wzbudzenia_sas, "sas", make_plots=False)

feats_ann = {"PSD" : widma,"DWT" : wavelet_trans_all}
feats_cnn = {"STFT" : stft_amplitudes, "MFCC" : mfccs_all}

feats_ann_sas = {"PSD" : widma_sas,"DWT" : wavelet_trans_all_sas}
feats_cnn_sas = {"STFT" : stft_amplitudes_sas, "MFCC" : mfccs_all_sas}

values, counts = np.unique(rodzaj_klasyfikacja, return_counts=True)
data_len = np.shape(rodzaj_klasyfikacja)[0]
class_weight = {0: 1/counts[0]*(data_len/2), 1:1/counts[1]*(data_len/2)}
print(counts)

values_sas, counts_sas = np.unique(rodzaj_klasyfikacja_sas, return_counts=True)
data_len_sas = np.shape(rodzaj_klasyfikacja_sas)[0]
class_weight_sas = {0: 1/counts_sas[0]*(data_len_sas/2), 1:1/counts_sas[1]*(data_len_sas/2)}
print(counts_sas)

#ANNs Chirp

accuracies = {}
for variant in feats_ann.keys():
    feat = feats_ann[variant]

    x_train, x_test, y_train, y_test = train_test_split(feat, np.asarray(rodzaj_klasyfikacja), test_size=0.3, stratify=rodzaj_klasyfikacja, random_state=random_state)
    
    model_ANN = make_model_ANN(input_shape=np.shape(x_train)[1])
    model_ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model_ANN.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight)
    accuracies[variant] = make_plots_history(model_ANN, "chirp", x_test, y_test, history, f"ANN_{variant}")

# # K-FOLD VALIDATE ANNs chirp
   
print("K-FOLD VALIDATING ANNs chirp")
for variant in feats_ann.keys():
    print(f"{variant} chirp")
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    feat = feats_ann[variant]
    for train, test in kfold.split(feat, np.asarray(rodzaj_klasyfikacja)):
        x_train = feat[train]
        x_test = feat[test]
        y_train = np.asarray(rodzaj_klasyfikacja)[train]
        y_test = np.asarray(rodzaj_klasyfikacja)[test]

        model_temp = make_model_ANN(input_shape=np.shape(x_train)[1])
        model_temp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model_temp.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight, verbose=0)
        scores = model_temp.evaluate(x_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model_temp.metrics_names[0]} of {scores[0]}; {model_temp.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    export_data_to_excel(loss_per_fold, "chirp", acc_per_fold, variant)


#ANNs SAS

accuracies = {}
for variant in feats_ann_sas.keys():
    feat = feats_ann_sas[variant]

    x_train, x_test, y_train, y_test = train_test_split(feat, np.asarray(rodzaj_klasyfikacja_sas), test_size=0.3, stratify=rodzaj_klasyfikacja_sas, random_state=random_state)
    
    model_ANN = make_model_ANN(input_shape=np.shape(x_train)[1])
    model_ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model_ANN.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight_sas)
    accuracies[variant] = make_plots_history(model_ANN, "sas", x_test, y_test, history, f"ANN_{variant}")

# # K-FOLD VALIDATE ANNs SAS
   
print("K-FOLD VALIDATING ANNs SAS")
for variant in feats_ann_sas.keys():
    print(f"{variant} sas")
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    feat = feats_ann_sas[variant]
    for train, test in kfold.split(feat, np.asarray(rodzaj_klasyfikacja_sas)):
        x_train = feat[train]
        x_test = feat[test]
        y_train = np.asarray(rodzaj_klasyfikacja_sas)[train]
        y_test = np.asarray(rodzaj_klasyfikacja_sas)[test]

        model_temp = make_model_ANN(input_shape=np.shape(x_train)[1])
        model_temp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model_temp.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight_sas, verbose=0)
        scores = model_temp.evaluate(x_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model_temp.metrics_names[0]} of {scores[0]}; {model_temp.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    export_data_to_excel(loss_per_fold, "sas", acc_per_fold, variant)

CNNs chirp

accuracies={}
for variant in feats_cnn.keys():
    feat = feats_cnn[variant]

    x_train, x_test, y_train, y_test = train_test_split(feat, np.asarray(rodzaj_klasyfikacja), test_size=0.3, stratify=rodzaj_klasyfikacja, random_state=random_state)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


    model_CNN = make_model_CNN(x_train.shape[1], x_train.shape[2])
    model_CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model_CNN.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.3, class_weight=class_weight)
    accuracies[variant] = make_plots_history(model_CNN, "chirp", x_test, y_test, history, f"CNN_{variant}")
    
print("K-FOLD VALIDATING CNNs chirp")
for variant in feats_cnn.keys():
    print(variant)
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    feat = feats_cnn[variant]
    for train, test in kfold.split(feat, np.asarray(rodzaj_klasyfikacja)):
        x_train = feat[train]
        x_test = feat[test]
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        y_train = np.asarray(rodzaj_klasyfikacja)[train]
        y_test = np.asarray(rodzaj_klasyfikacja)[test]

        model_temp = make_model_CNN(x_train.shape[1], x_train.shape[2])
        model_temp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model_temp.fit(x_train, y_train, epochs=15, batch_size=4, validation_split=0.3, class_weight=class_weight, verbose=0)
        scores = model_temp.evaluate(x_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model_temp.metrics_names[0]} of {scores[0]}; {model_temp.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    export_data_to_excel(loss_per_fold, "chirp", acc_per_fold, variant)

# CNNs sas

accuracies={}
for variant in feats_cnn_sas.keys():
    feat = feats_cnn_sas[variant]

    x_train, x_test, y_train, y_test = train_test_split(feat, np.asarray(rodzaj_klasyfikacja_sas), test_size=0.3, stratify=rodzaj_klasyfikacja_sas, random_state=random_state)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


    model_CNN = make_model_CNN(x_train.shape[1], x_train.shape[2])
    model_CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model_CNN.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.3, class_weight=class_weight_sas)
    accuracies[variant] = make_plots_history(model_CNN, "chirp", x_test, y_test, history, f"CNN_{variant}")
    
print("K-FOLD VALIDATING CNNs chirp")
for variant in feats_cnn_sas.keys():
    print(variant)
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    feat = feats_cnn_sas[variant]
    for train, test in kfold.split(feat, np.asarray(rodzaj_klasyfikacja_sas)):
        x_train = feat[train]
        x_test = feat[test]
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        y_train = np.asarray(rodzaj_klasyfikacja)[train]
        y_test = np.asarray(rodzaj_klasyfikacja)[test]

        model_temp = make_model_CNN(x_train.shape[1], x_train.shape[2])
        model_temp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model_temp.fit(x_train, y_train, epochs=15, batch_size=4, validation_split=0.3, class_weight=class_weight_sas, verbose=0)
        scores = model_temp.evaluate(x_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model_temp.metrics_names[0]} of {scores[0]}; {model_temp.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    export_data_to_excel(loss_per_fold, "sas", acc_per_fold, variant)


# PCA FOR ANNs chirp
accuracies_pca = {}
for variant in feats_ann.keys():
    feat = feats_ann[variant]

    feat_reduced = perform_PCA_reduction(feat, "chirp", f"{variant}_PCA", 10)
    x_train, x_test, y_train, y_test = train_test_split(feat, np.asarray(rodzaj_klasyfikacja), test_size=0.3, stratify=rodzaj_klasyfikacja, random_state=random_state)
    
    model_ANN = make_model_ANN(input_shape=np.shape(x_train)[1])
    model_ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model_ANN.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight)
    accuracies_pca[variant] = make_plots_history(model_ANN, "chirp", x_test, y_test, history, f"ANN_{variant}_PCA")

print("K-FOLD VALIDATING PCA")
for variant in feats_ann.keys():
    print(variant)
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    feat = feats_ann[variant]
    for train, test in kfold.split(feat, np.asarray(rodzaj_klasyfikacja)):
        feat_reduced = perform_PCA_reduction(feat, "chirp", f"{variant}_PCA", 10, do_plots=False)
        x_train = feat_reduced[train]
        x_test = feat_reduced[test]
        y_train = np.asarray(rodzaj_klasyfikacja)[train]
        y_test = np.asarray(rodzaj_klasyfikacja)[test]

        model_temp = make_model_ANN(input_shape=np.shape(x_train)[1])
        model_temp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model_temp.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight, verbose=0)
        scores = model_temp.evaluate(x_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model_temp.metrics_names[0]} of {scores[0]}; {model_temp.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    export_data_to_excel(loss_per_fold, "chirp", acc_per_fold, f"PCA_{variant}")

# PCA FOR ANNs SAS
accuracies_pca = {}
for variant in feats_ann_sas.keys():
    feat = feats_ann_sas[variant]

    feat_reduced = perform_PCA_reduction(feat, "sas", f"{variant}_PCA", 10)
    x_train, x_test, y_train, y_test = train_test_split(feat, np.asarray(rodzaj_klasyfikacja_sas), test_size=0.3, stratify=rodzaj_klasyfikacja_sas, random_state=random_state)
    
    model_ANN = make_model_ANN(input_shape=np.shape(x_train)[1])
    model_ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model_ANN.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight_sas)
    accuracies_pca[variant] = make_plots_history(model_ANN, "sas", x_test, y_test, history, f"ANN_{variant}_PCA")

print("K-FOLD VALIDATING PCA")
for variant in feats_ann_sas.keys():
    print(variant)
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    feat = feats_ann_sas[variant]
    for train, test in kfold.split(feat, np.asarray(rodzaj_klasyfikacja_sas)):
        feat_reduced = perform_PCA_reduction(feat, "sas", f"{variant}_PCA", 10, do_plots=False)
        x_train = feat_reduced[train]
        x_test = feat_reduced[test]
        y_train = np.asarray(rodzaj_klasyfikacja_sas)[train]
        y_test = np.asarray(rodzaj_klasyfikacja_sas)[test]

        model_temp = make_model_ANN(input_shape=np.shape(x_train)[1])
        model_temp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model_temp.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, class_weight=class_weight, verbose=0)
        scores = model_temp.evaluate(x_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model_temp.metrics_names[0]} of {scores[0]}; {model_temp.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    export_data_to_excel(loss_per_fold, "sas", acc_per_fold, f"PCA_{variant}")