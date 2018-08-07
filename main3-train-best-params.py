import pickle
import numpy as np
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

# load data
train_pi_labels_onehot = np.load('/mnt/train_pi_labels_onehot.out.npy')
train_dna_seqs = pickle.load(open('/mnt/train_dna_seqs.out', 'rb'))
cl_weight = pickle.load(open('/mnt/class_weight.out', 'rb'))
val_pi_labels_onehot = np.load('/mnt/val_pi_labels_onehot.out.npy')
val_dna_seqs = pickle.load(open('/mnt/val_dna_seqs.out', 'rb'))
val_dna_seqs_onehot = np.transpose(convert_onehot2D(val_dna_seqs), axes=(0,2,1))
num_classes = train_pi_labels_onehot.shape[1]
dna_bp_length = len(train_dna_seqs[0])
print("Shape of training data =", len(train_dna_seqs), "sequences of", dna_bp_length, "bp")

# network hyperparameters
filter_num = 128 
filter_len = 12
num_dense_nodes = 64 
total_epoch = 100 
min_batch_size = 8

# model specification
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="checkpoint.hdf5", monitor="val_acc", mode='auto', verbose=1, \
              save_best_only=True)  
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, mode='auto', verbose=1)
print(model.summary())

num_chunks = 6 # number of chunks the data is split into

# initialize
max_val_acc = 0.0
max_acc_pair = 0.0

# store the train/val accuracy and loss for each epoch
epoch_train_acc = np.zeros((total_epoch,num_chunks))
epoch_val_acc = np.zeros((total_epoch,1))
epoch_train_loss = np.zeros((total_epoch,num_chunks))
epoch_val_loss = np.zeros((total_epoch,1))

# train the model
for epoch in range(total_epoch):
    print("Epoch =", epoch+1, "out of", total_epoch)
    for f in range(num_chunks-1):
        X_train = np.load("/mnt/data"+str(f)+".npy")
        y_train = np.load("/mnt/labels"+str(f)+".npy")
        history = model.fit(X_train, y_train, batch_size = min_batch_size, \
              validation_split=0.0, nb_epoch=1, verbose=1, class_weight=cl_weight)
        
        # store training stats
        epoch_train_acc[epoch,f] = history.history['acc'][0]
        epoch_train_loss[epoch,f] = history.history['loss'][0]

    # train final chunk and do validation
    X_train = np.load("/mnt/data"+str(num_chunks-1)+".npy")
    y_train = np.load("/mnt/labels"+str(num_chunks-1)+".npy")  
    history = model.fit(X_train, y_train, batch_size = min_batch_size, \
              validation_data=(val_dna_seqs_onehot, val_pi_labels_onehot), nb_epoch=1, verbose=1, class_weight=cl_weight, \
              callbacks=[checkpointer,early_stopping])
    
    # store training and validation stats
    epoch_train_acc[epoch,num_chunks-1] = history.history['acc'][0]
    epoch_val_acc[epoch,0] = history.history['val_acc'][0]
    epoch_train_loss[epoch,num_chunks-1] = history.history['loss'][0]
    epoch_val_loss[epoch,0] = history.history['val_loss'][0]
    
# serialize model to JSON and save training stats
model.save('best_model.h5') 
np.save('best_model_train_acc.out', np.transpose(np.mean(epoch_train_acc, axis=1)))
np.save('best_model_train_loss.out', np.transpose(np.mean(epoch_train_loss, axis=1)))
np.save('best_model_val_acc.out', epoch_val_acc)
np.save('best_model_val_loss.out', epoch_val_loss)

