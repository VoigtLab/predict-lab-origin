import pickle
import csv
import numpy as np
from numpy import linalg as LA
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from utils_EC2 import *
from keras.layers.normalization import BatchNormalization
from keras.models import load_model 

# get JSON tree of plasmids
addgene_json_plasmids = get_json_plasmids('addgene-plasmids-sequences.json')

# get dict of PI:plasmid counts above threshold
min_num_plasmids_cutoff = 9
max_seq_length = 8000
pi_plasmid_dict = get_num_plasmids_per_pi(addgene_json_plasmids, min_num_plasmids_cutoff, max_seq_length)
cnn_filter_length = 48
disruption_len = 50
[pi_labels, dna_seqs, annotation_labels, plasmid_names] = get_seqs_annotations(addgene_json_plasmids, pi_plasmid_dict, cnn_filter_length, max_seq_length)
plasmid_of_interest = []
label_of_interest = []
dna_seq_of_interest = []

# load in 4 plasmids of interest and depositing labs
plasmid_of_interest.append('pCI-YFP')
label_of_interest.append(745)
dna_seq_of_interest.append('TACTAGTAGCGGCCGCTGCAGTCCGGCAAAAAAACGGGCAAGGTGTCACCACCCTGCCCTTTTTCTTTAAAACCGAAAAGATTACTTCGCGTTATGCAGGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATCTCGAGTCCCGTCAAGTCAGCGTAATGCTCTGCCAGTGTTACAACCAATTAACCAATTCTGATTAGAAAAACTCATCGAGCATCAAATGAAACTGCAATTTATTCATATCAGGATTATCAATACCATATTTTTGAAAAAGCCGTTTCTGTAATGAAGGAGAAAACTCACCGAGGCAGTTCCATAGGATGGCAAGATCCTGGTATCGGTCTGCGATTCCGACTCGTCCAACATCAATACAACCTATTAATTTCCCCTCGTCAAAAATAAGGTTATCAAGTGAGAAATCACCATGAGTGACGACTGAATCCGGTGAGAATGGCAAAAGCTTATGCATTTCTTTCCAGACTTGTTCAACAGGCCAGCCATTACGCTCGTCATCAAAATCACTCGCATCAACCAAACCGTTATTCATTCGTGATTGCGCCTGAGCGAGACGAAATACGCGATCGCTGTTAAAAGGACAATTACAAACAGGAATCGAATGCAACCGGCGCAGGAACACTGCCAGCGCATCAACAATATTTTCACCTGAATCAGGATATTCTTCTAATACCTGGAATGCTGTTTTCCCGGGGATCGCAGTGGTGAGTAACCATGCATCATCAGGAGTACGGATAAAATGCTTGATGGTCGGAAGAGGCATAAATTCCGTCAGCCAGTTTAGTCTGACCATCTCATCTGTAACATCATTGGCAACGCTACCTTTGCCATGTTTCAGAAACAACTCTGGCGCATCGGGCTTCCCATACAATCGATAGATTGTCGCACCTGATTGCCCGACATTATCGCGAGCCCATTTATACCCATATAAATCAGCATCCATGTTGGAATTTAATCGCGGCCTCGAGCAAGACGTTTCCCGTTGAATATGGCTCATAACACCCCTTGTATTACTGTTTATGTAAGCAGACAGTTTTATTGTTCATGATGATATATTTTTATCTTGTGCAATGTAACATCAGAGATTTTGAGACACAACGTGGCTTTGTTGAATAAATCGAACTTTTGCTGAGTTGAAGGATCAGATCACGCATCTTCCCGACAACGCAGACCGTTCCGTGGCAAAGCAAAAGTTCAAAATCACCAACTGGTCCACCTACAACAAAGCTCTCATCAACCGTGGCTCCCTCACTTTCTGGCTGGATGATGGGGCGATTCAGGCCTGGTATGAGTCAGCAACACCTTCTTCACGAGGCAGACCTCAGCGCTAGCGGAGTGTATACTGGCTTACTATGTTGGCACTGATGAGGGTGTCAGTGAAGTGCTTCATGTGGCAGGAGAAAAAAGGCTGCACCGGTGCGTCAGCAGAATATGTGATACAGGATATATTCCGCTTCCTCGCTCACTGACTCGCTACGCTCGGTCGTTCGACTGCGGCGAGCGGAAATGGCTTACGAACGGGGCGGAGATTTCCTGGAAGATGCCAGGAAGATACTTAACAGGGAAGTGAGAGGGCCGCGGCAAAGCCGTTTTTCCATAGGCTCCGCCCCCCTGACAAGCATCACGAAATCTGACGCTCAAATCAGTGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCTGGCGGCTCCCTCGTGCGCTCTCCTGTTCCTGCCTTTCGGTTTACCGGTGTCATTCCGCTGTTATGGCCGCGTTTGTCTCATTCCACGCCTGACACTCAGTTCCGGGTAGGCAGTTCGCTCCAAGCTGGACTGTATGCACGAACCCCCCGTTCAGTCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGAAAGACATGCAAAAGCACCACTGGCAGCAGCCACTGGTAATTGATTTAGAGGAGTTAGTCTTGAAGTCATGCGCCGGTTAAGGCTAAACTGAAAGGACAAGTTTTGGTGACTGCGCTCCTCCAAGCCAGTTACCTCGGTTCAAAGAGTTGGTAGCTCAGAGAACCTTCGAAAAACCGCCCTGCAAGGCGGTTTTTTCGTTTTCAGAGCAAGAGATTACGCGCAGACCAAAACGATCTCAAGAAGATCATCTTATTAAGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGTTACCAATGCTTAATCAGTGAGGCACCTATCTCAGCGATCTGTCTATTTCGTTCATCCATAGTTGCCTGACTCCCCGTCGTGTAGATAACTACGATACGGGAGGGCTTACCATCTGGCCCCAGTGCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAAGGGCCGAGCGCAGAAGTGGTCCTGCAACTTTATCCGCCTCCATCCAGTCTATTCCATGGTGCCACCTGACGTCTAAGAAACCATTATTATCATGACATTAACCTATAAAAATAGGCGTATCACGAGGCAGAATTTCAGATAAAAAAAATCCTTAGCTTTCGCTAAGGATGATTTCTGGAATTCGCGGCCGCTTCTAGAGTAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGCTACTAGAGAAAGAGGAGAAATACTAGATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGCAATGCTTCGCCCGCTACCCCGACCACATGAAGCTGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAATAATACTAGAGCCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATA')
plasmid_of_interest.append('Flag-HA-BRCC3')
label_of_interest.append(273)
plasmid_of_interest.append('pT7CFE1-TelR15-YPet')
label_of_interest.append(532)
plasmid_of_interest.append('pAAV-CAG-tdTomato (codon diversified)')
label_of_interest.append(62)

# load in DNA seqs of interest
for i in range(len(plasmid_names)):
    if plasmid_names[i] in plasmid_of_interest:
        print(plasmid_names[i])
        dna_seq_of_interest.append(dna_seqs[i][:-cnn_filter_length])
        print(len(dna_seq_of_interest[-1]))

undisrupted_seqs = []
for a in dna_seq_of_interest:
    undisrupted_seqs.append(a+'N'*cnn_filter_length)
undisrupted_seqs = pad_dna(undisrupted_seqs,max_seq_length)
undisrupted_seqs = append_rc(undisrupted_seqs,cnn_filter_length)
undisrupted_dna_seqs_onehot = np.transpose(convert_onehot2D(undisrupted_seqs), axes=(0,2,1))

# load model
sorted_pi_list = np.load('/mnt/sorted_pi_list.out')
model = load_model('best_model.h5')
print("Loaded model")
print(model.summary())

# make model that ends at last neuron pre-softmax
c1 = model.get_layer("conv1d_1")
b1 = model.get_layer("batch_normalization_1")
d1 = model.get_layer("dense_1")
b2 = model.get_layer("batch_normalization_2")
d2 = model.get_layer("dense_2")
model2 = Sequential()
model2.add(Convolution1D(input_shape=(16048,4), filters=128, kernel_size=12, activation="relu", padding="same", weights=c1.get_weights()))
model2.add(MaxPooling1D(pool_size=16048))
model2.add(BatchNormalization(weights=b1.get_weights()))
model2.add(Flatten())
model2.add(Dense(input_dim=128,units=64, weights=d1.get_weights()))
model2.add(Activation("relu"))
model2.add(BatchNormalization(weights=b2.get_weights()))
model2.add(Dense(units=827, weights=d2.get_weights()))
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

# cycle through each plasmid of interest
for h in range(len(dna_seq_of_interest)):
    seq = dna_seq_of_interest[h]
    
    # intialize variables for plasmid disruption scan
    total_plasmid_size = len(seq)+disruption_len
    probability_for_correct = np.zeros(total_plasmid_size)
    probability_for_1st = np.zeros(total_plasmid_size)
    probability_for_2nd = np.zeros(total_plasmid_size)
    probability_for_3rd = np.zeros(total_plasmid_size)
    probability_for_4th = np.zeros(total_plasmid_size)
    probability_for_5th = np.zeros(total_plasmid_size)
    probability_for_6th = np.zeros(total_plasmid_size)
    probability_for_7th = np.zeros(total_plasmid_size)

    outneuron_for_correct = np.zeros(total_plasmid_size)
    outneuron_for_1st = np.zeros(total_plasmid_size)
    outneuron_for_2nd = np.zeros(total_plasmid_size)
    outneuron_for_3rd = np.zeros(total_plasmid_size)
    outneuron_for_4th = np.zeros(total_plasmid_size)
    outneuron_for_5th = np.zeros(total_plasmid_size)
    outneuron_for_6th = np.zeros(total_plasmid_size)
    outneuron_for_7th = np.zeros(total_plasmid_size)

    prob_geomean = np.zeros(total_plasmid_size)
    prob_std_log = np.zeros(total_plasmid_size)
    out_mean = np.zeros(total_plasmid_size)
    out_std = np.zeros(total_plasmid_size)
    rank_prediction = np.zeros(total_plasmid_size)
    top_lab = np.zeros(total_plasmid_size)
    
    predicted_vector = model.predict(np.expand_dims(undisrupted_dna_seqs_onehot[h], axis=0),verbose=0)[0]
    predicted_index = np.argmax(predicted_vector)
    actual_index = label_of_interest[h]
    sorted_indices = np.argsort(predicted_vector)
    max_l2_correct_dna_seq = dna_seq_of_interest[h]
    max_l2_correct_dna_seq_onehot = undisrupted_dna_seqs_onehot[h]
    max_l2_correct_actual_index = actual_index
    max_l2_correct_predicted_index_1st = sorted_indices[-1]
    max_l2_correct_predicted_index_2nd = sorted_indices[-2]
    max_l2_correct_predicted_index_3rd = sorted_indices[-3]
    max_l2_correct_predicted_index_4th = sorted_indices[-4]
    max_l2_correct_predicted_index_5th = sorted_indices[-5]
    max_l2_correct_predicted_index_6th = sorted_indices[-6]
    max_l2_correct_predicted_index_7th = sorted_indices[-7]
    
    # disrupt each window of DNA
    for i in range(len(seq)):
        if i <= len(seq) - disruption_len:
            disrupted_seq = seq[:i] + 'N'*disruption_len + seq[i+disruption_len:]
        else:
            front_disrupt_stop = i-(len(seq)-disruption_len)
            back_disrupt_begin = front_disrupt_stop - disruption_len 
            disrupted_seq = 'N'*front_disrupt_stop + seq[front_disrupt_stop:back_disrupt_begin] + -back_disrupt_begin*'N'
        
        # mask sequence outside of filter-sized window
        disrupted_seq = pad_dna([disrupted_seq+'N'*cnn_filter_length, ''],max_seq_length)
        disrupted_seq = append_rc(disrupted_seq,cnn_filter_length)
        total_plasmid_size = len(disrupted_seq)
        disrupted_seq_onehot = np.transpose(convert_onehot2D(disrupted_seq), axes=(0,2,1))
        
        # predict softmax and neural activities across all labs
        prob_vector = model.predict(np.expand_dims(disrupted_seq_onehot[0], axis=0),verbose=0)[0]
        out_vector = model2.predict(np.expand_dims(disrupted_seq_onehot[0], axis=0),verbose=0)[0] #output neuron no softmax
        prob_vector_log = np.log10(prob_vector)
        prob_geomean[i] = 10**np.mean(prob_vector_log)
        prob_std_log[i] = np.std(prob_vector_log)
        out_mean[i] = np.mean(out_vector)
        out_std[i] = np.std(out_vector)

        # compute what rank correct prediction is
        # top ranking = 1; bottom = 827
        top_lab[i] = np.argmax(prob_vector)

        probability_for_correct[i:i+disruption_len] += prob_vector[max_l2_correct_actual_index]        
        probability_for_1st[i:i+disruption_len] += prob_vector[max_l2_correct_predicted_index_1st]
        probability_for_2nd[i:i+disruption_len] += prob_vector[max_l2_correct_predicted_index_2nd]
        probability_for_3rd[i:i+disruption_len] += prob_vector[max_l2_correct_predicted_index_3rd]
        probability_for_4th[i:i+disruption_len] += prob_vector[max_l2_correct_predicted_index_4th]
        probability_for_5th[i:i+disruption_len] += prob_vector[max_l2_correct_predicted_index_5th]
        probability_for_6th[i:i+disruption_len] += prob_vector[max_l2_correct_predicted_index_6th]
        probability_for_7th[i:i+disruption_len] += prob_vector[max_l2_correct_predicted_index_7th]

        outneuron_for_correct[i:i+disruption_len] += out_vector[max_l2_correct_actual_index]
        outneuron_for_1st[i:i+disruption_len] += out_vector[max_l2_correct_predicted_index_1st]
        outneuron_for_2nd[i:i+disruption_len] += out_vector[max_l2_correct_predicted_index_2nd]
        outneuron_for_3rd[i:i+disruption_len] += out_vector[max_l2_correct_predicted_index_3rd]
        outneuron_for_4th[i:i+disruption_len] += out_vector[max_l2_correct_predicted_index_4th]
        outneuron_for_5th[i:i+disruption_len] += out_vector[max_l2_correct_predicted_index_5th]
        outneuron_for_6th[i:i+disruption_len] += out_vector[max_l2_correct_predicted_index_6th]
        outneuron_for_7th[i:i+disruption_len] += out_vector[max_l2_correct_predicted_index_7th]  

    # add last 50 to first 50 to make circular
    probability_for_correct[:disruption_len] += probability_for_correct[-disruption_len:] 
    probability_for_1st[:disruption_len] += probability_for_1st[-disruption_len:] 
    probability_for_2nd[:disruption_len] += probability_for_2nd[-disruption_len:] 
    probability_for_3rd[:disruption_len] += probability_for_3rd[-disruption_len:] 
    probability_for_4th[:disruption_len] += probability_for_4th[-disruption_len:] 
    probability_for_5th[:disruption_len] += probability_for_5th[-disruption_len:] 
    probability_for_6th[:disruption_len] += probability_for_6th[-disruption_len:] 
    probability_for_7th[:disruption_len] += probability_for_7th[-disruption_len:] 

    outneuron_for_correct[:disruption_len] += outneuron_for_correct[-disruption_len:] 
    outneuron_for_1st[:disruption_len] += outneuron_for_1st[-disruption_len:] 
    outneuron_for_2nd[:disruption_len] += outneuron_for_2nd[-disruption_len:] 
    outneuron_for_3rd[:disruption_len] += outneuron_for_3rd[-disruption_len:] 
    outneuron_for_4th[:disruption_len] += outneuron_for_4th[-disruption_len:] 
    outneuron_for_5th[:disruption_len] += outneuron_for_5th[-disruption_len:] 
    outneuron_for_6th[:disruption_len] += outneuron_for_6th[-disruption_len:] 
    outneuron_for_7th[:disruption_len] += outneuron_for_7th[-disruption_len:]
    
    # write output file
    np.savetxt(str(h) + '_disruption_scan_' + sorted_pi_list[max_l2_correct_actual_index] + "_" \
                  + plasmid_of_interest[h].replace("/","-").replace(" ","") + '.out', \
               (top_lab, outneuron_for_1st/50, outneuron_for_2nd/50, outneuron_for_3rd/50, outneuron_for_4th/50, \
               outneuron_for_5th/50, outneuron_for_6th/50, outneuron_for_7th/50), delimiter=',')
