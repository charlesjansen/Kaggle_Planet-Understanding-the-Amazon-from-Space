import numpy as np 
import pandas as pd 
import os
import cv2
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score

x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('../input/train_v2.csv')
df_test = pd.read_csv('../input/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

img_size = 64
channels = 4 #4 for tiff, 3 for jpeg
for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('../input/test-tif-v2/{}.tif'.format(f), -1)
    x_test.append(cv2.resize(img, (img_size, img_size)))
x_test  = np.array(x_test, np.float32)/255. 

for f, tags in tqdm(df_train.values, miniters=1000):
    #https://stackoverflow.com/questions/37512119/resize-transparent-image-in-opencv-python-cv2
    #If you load a 4 channel image, the flag -1 indicates that the image is loaded unchanged, so you can load and split all 4 channels directly.
    img = cv2.imread('../input/train-tif-v2/{}.tif'.format(f), -1)#0-1 voir au dessus les 2 comments
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (img_size, img_size)))
    y_train.append(targets)
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)/255.


#use saved variable to start from here


print(x_train.shape)
print(y_train.shape)

nfolds = 2

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train =[]

kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1234)

for train_index, test_index in kf:
       
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        
        model = Sequential()
        model.add(BatchNormalization(input_shape=(img_size, img_size, channels)))
        model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(512, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(17, activation='sigmoid'))

        epochs_arr  = [   20,      5,    5,       5]
        learn_rates = [0.01, 0.001, 0.0001, 0.00001]

        for learn_rate, epochs in zip(learn_rates, epochs_arr):
            opt  = optimizers.Adam(lr=learn_rate)
            model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])
            callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                         ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=1)]

            model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=256, verbose=1, epochs=epochs, callbacks=callbacks, shuffle=True)
        
        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)
        
        
        p_valid = model.predict(X_valid, batch_size = 128, verbose=1)
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

        p_train = model.predict(x_train, batch_size =128, verbose=1)
        yfull_train.append(p_train)
        
        p_test = model.predict(x_test, batch_size = 128, verbose=1)
        yfull_test.append(p_test)

result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)
result

from tqdm import tqdm
thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))
    
df_test['tags'] = preds
df_test.to_csv('F:/DS-main/Kaggle-main/Planet Understanding the Amazon from Space/submission_keras_1morelater_allBN_2fold_tiff_64size128batch.csv', index=False)

## 0.913