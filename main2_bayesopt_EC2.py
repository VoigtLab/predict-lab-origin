# BAYESIAN OPTIMIZATION ON CONV-DENSE 

import pickle
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from utils_EC2 import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import time
from keras.optimizers import SGD
from bayes_opt import BayesianOptimization

global train_pi_labels_onehot
global train_dna_seqs
global cl_weight
global val_pi_labels_onehot
global val_dna_seqs_onehot
global dna_bp_length

def target(total_epoch, filter_num, filter_len, num_dense_nodes):
    
    start = time.time()
    
    total_epoch = int(round(total_epoch))
    filter_num = int(round(filter_num))
    filter_len = int(round(filter_len))
    num_dense_nodes = int(round(num_dense_nodes))
    print("Epochs =", total_epoch, "| # Conv filters =", filter_num, "| Filter length =", filter_len, "| # Dense nodes =", num_dense_nodes)

    model = Sequential()
    model.add(Convolution1D(input_dim=4, input_length=dna_bp_length, nb_filter=filter_num, filter_length=filter_len, activation="relu", border_mode ="same"))
    model.add(MaxPooling1D(pool_length=dna_bp_length))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(input_dim=filter_num,output_dim=num_dense_nodes))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dense(output_dim=num_classes))
    model.add(Activation("softmax"))

    print("Compile")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    print(model.summary())
  
    max_val_acc = 0.0
    max_acc_pair = 0.0
    num_chunks = 6  
    
    epoch_train_acc = np.zeros((total_epoch,num_chunks))
    epoch_val_acc = np.zeros((total_epoch,1))

    for e in range(total_epoch):
        print("Epoch =", e+1, "out of", total_epoch)
        for f in range(num_chunks-1):
            X_train = np.load("/mnt/data"+str(f)+".npy")
            y_train = np.load("/mnt/labels"+str(f)+".npy")
            history = model.fit(X_train, y_train, batch_size = 8, \
                  validation_split=0.0, nb_epoch=1, verbose=1, class_weight=cl_weight)
            epoch_train_acc[e,f] = history.history['acc'][0]
            
        X_train = np.load("/mnt/data"+str(num_chunks-1)+".npy")
        y_train = np.load("/mnt/labels"+str(num_chunks-1)+".npy")
        history = model.fit(X_train, y_train, batch_size = 8, \
                  validation_data=(val_dna_seqs_onehot, val_pi_labels_onehot), nb_epoch=1, verbose=1, class_weight=cl_weight)
        epoch_train_acc[e,num_chunks-1] = history.history['acc'][0]
        epoch_val_acc[e,0] = history.history['val_acc'][0]

        if history.history['val_acc'][0] > max_val_acc:
            max_val_acc = history.history['val_acc'][0]
            max_acc_pair = history.history['acc'][0]
    
    
    print("Epoch training accuracy")
    print(epoch_train_acc)
    print("Mean epoch training accuracy")
    print(np.transpose(np.mean(epoch_train_acc, axis=1)))

    end = time.time()
    
    np.save(str(int(end))+'conv'+str(filter_num)+'x'+str(filter_len)+'dense'+str(num_dense_nodes)+'time'+str(int(end-start))+'_mean_train_acc.out', np.transpose(np.mean(epoch_train_acc, axis=1)))
    print("Epoch validation accuracy" )
    print(epoch_val_acc)

    np.save(str(int(end))+'conv'+str(filter_num)+'x'+str(filter_len)+'dense'+str(num_dense_nodes)+'time'+str(int(end-start))+'_epoch_val_acc.out', epoch_val_acc, end-start)
     
    return max_val_acc/(end-start)


# SAVE BEST VAL_ACC, ACC, ASSOCIATED NN WEIGHTS, AND BO PARAMS

# Load data
timestamp = '1497248061'

train_pi_labels_onehot = np.load('/mnt/' + timestamp + '_train_pi_labels_onehot.out.npy')
train_dna_seqs = pickle.load(open('/mnt/' + timestamp + '_train_dna_seqs.out', 'rb'))
cl_weight = pickle.load(open('/mnt/' + timestamp + '_class_weight.out', 'rb'))
val_pi_labels_onehot = np.load('/mnt/' + timestamp + '_val_pi_labels_onehot.out.npy')
val_dna_seqs = pickle.load(open('/mnt/' + timestamp + '_val_dna_seqs.out', 'rb'))
val_dna_seqs_onehot = np.transpose(convert_onehot2D(val_dna_seqs), axes=(0,2,1))

global num_classes
num_classes = val_pi_labels_onehot.shape[1]
global dna_bp_length
dna_bp_length = len(val_dna_seqs[0])

print("Start Bayesian optimization")
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
bo = BayesianOptimization(target, {'total_epoch': (5,5), 'filter_num': (1, 512), 'filter_len': (1, 48), 'num_dense_nodes': (1, 256)})
bo.explore({'total_epoch': [5,5,5], 'filter_num': [512,256,128], 'filter_len': [48,24,12], 'num_dense_nodes': [256,128,64]})
bo.maximize(init_points=0, n_iter=20, acq="ucb", kappa=5, **gp_params)

# The output values can be accessed with self.res
print(bo.res['max'])
print(bo.res['all'])

