# AG DATASET CNN, DNA TO PI 

import pickle
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from utils_EC2 import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import time
from keras.optimizers import SGD
from keras.models import load_model 

# Best from bayesopt = 1497284722conv128x12dense64time3588_epoch_val_acc.out.npy


# Load data
timestamp = '1497248061'
train_pi_labels_onehot = np.load('/mnt/' + timestamp + '_train_pi_labels_onehot.out.npy')
train_dna_seqs = pickle.load(open('/mnt/' + timestamp + '_train_dna_seqs.out', 'rb'))
cl_weight = pickle.load(open('/mnt/' + timestamp + '_class_weight.out', 'rb'))
val_pi_labels_onehot = np.load('/mnt/' + timestamp + '_val_pi_labels_onehot.out.npy')
val_dna_seqs = pickle.load(open('/mnt/' + timestamp + '_val_dna_seqs.out', 'rb'))
val_dna_seqs_onehot = np.transpose(convert_onehot2D(val_dna_seqs), axes=(0,2,1))

num_classes = train_pi_labels_onehot.shape[1]
dna_bp_length = len(train_dna_seqs[0])

print("Shape of training data =", len(train_dna_seqs), "sequences of", dna_bp_length, "bp")

# 62.19% val_acc after 200 epochs
filter_num = 128 
filter_len = 12
num_dense_nodes = 64 
total_epoch = 100 
min_batch_size = 8

## Model specification
model = Sequential()
model.add(Convolution1D(input_dim=4, input_length=dna_bp_length, nb_filter=filter_num, filter_length=filter_len, \
                        activation="relu", border_mode ="same"))
model.add(MaxPooling1D(pool_length=dna_bp_length))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(input_dim=filter_num,output_dim=num_dense_nodes))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(output_dim=num_classes))
model.add(Activation("softmax"))


print("Compile")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="checkpoint.hdf5", monitor="val_acc", mode='auto', verbose=1, \
              save_best_only=True)  
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, mode='auto', verbose=1)

print(model.summary())
   
max_val_acc = 0.0
max_acc_pair = 0.0
num_chunks = 6    

epoch_train_acc = np.zeros((total_epoch,num_chunks))
epoch_val_acc = np.zeros((total_epoch,1))
epoch_train_loss = np.zeros((total_epoch,num_chunks))
epoch_val_loss = np.zeros((total_epoch,1))

for epoch in range(total_epoch):
    print("Epoch =", epoch+1, "out of", total_epoch)
    for f in range(num_chunks-1):
        X_train = np.load("/mnt/data"+str(f)+".npy")
        y_train = np.load("/mnt/labels"+str(f)+".npy")
        history = model.fit(X_train, y_train, batch_size = min_batch_size, \
              validation_split=0.0, nb_epoch=1, verbose=1, class_weight=cl_weight)
        
        epoch_train_acc[epoch,f] = history.history['acc'][0]
        epoch_train_loss[epoch,f] = history.history['loss'][0]

    X_train = np.load("/mnt/data"+str(num_chunks-1)+".npy")
    y_train = np.load("/mnt/labels"+str(num_chunks-1)+".npy")  
    history = model.fit(X_train, y_train, batch_size = min_batch_size, \
              validation_data=(val_dna_seqs_onehot, val_pi_labels_onehot), nb_epoch=1, verbose=1, class_weight=cl_weight, \
              callbacks=[checkpointer,early_stopping])
    
    epoch_train_acc[epoch,num_chunks-1] = history.history['acc'][0]
    epoch_val_acc[epoch,0] = history.history['val_acc'][0]
    epoch_train_loss[epoch,num_chunks-1] = history.history['loss'][0]
    epoch_val_loss[epoch,0] = history.history['val_loss'][0]
    
# serialize model to JSON
model.save('2017-06-14_model.h5')  # creates a HDF5 file 'my_model.h5'
print("Saved model to disk")
    
np.save('2017-06-14_model_train_acc.out', np.transpose(np.mean(epoch_train_acc, axis=1)))
np.save('2017-06-14_model_train_loss.out', np.transpose(np.mean(epoch_train_loss, axis=1)))
np.save('2017-06-14_model_val_acc.out', epoch_val_acc)
np.save('2017-06-14_model_val_loss.out', epoch_val_loss)

