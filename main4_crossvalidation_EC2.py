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



# Load data
timestamp = '1497248061'
val_pi_labels_onehot = np.load('/mnt/' + timestamp + '_val_pi_labels_onehot.out.npy')
val_dna_seqs = pickle.load(open('/mnt/' + timestamp + '_val_dna_seqs.out', 'rb'))
val_dna_seqs_onehot = np.transpose(convert_onehot2D(val_dna_seqs), axes=(0,2,1))
test_pi_labels = pickle.load(open('/mnt/' + timestamp + '_test_pi_labels.out', 'rb'))
test_pi_labels_onehot = np.load('/mnt/' + timestamp + '_test_pi_labels_onehot.out.npy')
test_dna_seqs = pickle.load(open('/mnt/' + timestamp + '_test_dna_seqs.out', 'rb'))
test_dna_seqs_onehot = np.transpose(convert_onehot2D(test_dna_seqs), axes=(0,2,1))

model = load_model('checkpoint.hdf5')
print("Loaded model")
print(model.summary())

#print("Compile")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

print("Validation accuracy")  
start = time.time()
val_scores = model.evaluate(val_dna_seqs_onehot, val_pi_labels_onehot, verbose = 1)
end = time.time()
print("Validation time = ", end-start)
print("%s: %.2f%%" % (model.metrics_names[1], val_scores[1]*100))

print("Testing accuracy")
start = time.time()
test_scores = model.evaluate(test_dna_seqs_onehot, test_pi_labels_onehot, verbose = 1)
end = time.time()
print("Testing time = ", end-start)
print("%s: %.2f%%" % (model.metrics_names[1], test_scores[1]*100))

ranking_per_test_plasmid = np.zeros(test_dna_seqs_onehot.shape[0])
lab_num_correct_test_plasmid = np.zeros(test_pi_labels_onehot.shape[1])


for i in range(test_dna_seqs_onehot.shape[0]):
    predicted_vector = model.predict(np.expand_dims(test_dna_seqs_onehot[i], axis=0),verbose=0)[0]
    actual_index = np.argmax(test_pi_labels_onehot[i]) 
    predicted_argsort = np.argsort(predicted_vector)
    ranking_per_test_plasmid[i] = (np.where(predicted_argsort==actual_index)[0][0] - 827)*-1
    if ranking_per_test_plasmid[i] == 1:
        lab_num_correct_test_plasmid[actual_index] += 1


all_predictions = []
start = time.time()
for j in range(test_dna_seqs_onehot.shape[0]):
    predicted_vector = model.predict(np.expand_dims(test_dna_seqs_onehot[j], axis=0),verbose=0)[0]
    all_predictions.append(np.argmax(predicted_vector))
end = time.time()
np.save('all_predictions.out', all_predictions)
print("Making predictions for full test set took " + str(end-start) + " seconds")

"""
print("Correct plasmids per lab")
for j in lab_num_correct_test_plasmid:
    print(j)

print("Per plasmid rankings")
for k in ranking_per_test_plasmid:
    print(k)
    
"""