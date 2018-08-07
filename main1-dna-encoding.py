from utils_EC2 import *
import pickle
import numpy as np
import time

# get JSON tree of plasmids
addgene_json_plasmids = get_json_plasmids('addgene-plasmids-sequences.json')

# get dict of PI:plasmid counts above threshold
min_num_plasmids_cutoff = 9
max_seq_length = 8000
pi_plasmid_dict = get_num_plasmids_per_pi(addgene_json_plasmids, min_num_plasmids_cutoff, max_seq_length)
cnn_filter_length = 48
[pi_labels, dna_seqs, annotation_labels, plasmid_name] = get_seqs_annotations(addgene_json_plasmids, pi_plasmid_dict, cnn_filter_length, max_seq_length)

lens = []
for j in dna_seqs:
    lens.append(len(j))
lens.sort()

# pad short sequences, append reverse complement, separated by Ns
dna_seqs = pad_dna(dna_seqs,max_seq_length)
dna_seqs = append_rc(dna_seqs,cnn_filter_length)

# randomly permute rows
[pi_labels, dna_seqs, annotation_labels] = permute_order(pi_labels, dna_seqs, annotation_labels)

# convert pi_labels to one-hot vector, annotations to bag of words 
[sorted_pi_names, pi_labels_onehot] = convert_pi_labels_onehot(pi_labels)
[sorted_annotations, annotation_labels_bow] = convert_annotations(annotation_labels)

# feature selection for annotation bag of words using L1-penalized SVC
# annotations not used for downstream training
C = 0.1
[sorted_reduced_annotations, reduced_annotation_labels_bow] = feature_selection(pi_labels_onehot, annotation_labels_bow, sorted_annotations, C)

# set aside validation set
val_plasmids_per_pi = 3
test_plasmids_per_pi = 3
timestamp = str(int(time.time()))
separate_train_val_test([min_num_plasmids_cutoff, cnn_filter_length, C, val_plasmids_per_pi, test_plasmids_per_pi], \
                        sorted_pi_names, sorted_annotations, sorted_reduced_annotations, \
                        pi_labels, pi_labels_onehot, dna_seqs, annotation_labels, annotation_labels_bow, reduced_annotation_labels_bow, \
                        timestamp)

# load data
train_pi_labels_onehot = np.load('/mnt/train_pi_labels_onehot.out.npy')
train_reduced_annotation_labels_bow = np.load('/mnt/train_reduced_annotation_labels_bow.out.npy')
cl_weight = pickle.load(open('/mnt/class_weight.out', 'rb'))
val_pi_labels_onehot = np.load('/mnt/val_pi_labels_onehot.out.npy')
val_reduced_annotation_labels_bow = np.load('/mnt/val_reduced_annotation_labels_bow.out.npy')
train_dna_seqs = pickle.load(open('/mnt/train_dna_seqs.out', 'rb'))
test_pi_labels_onehot = np.load('/mnt/test_pi_labels_onehot.out.npy')

# save training-data in chunks to fit into GPU memory
num_training_plasmids = len(train_dna_seqs)
num_chunks = 6
chunk_size = int(num_training_plasmids/num_chunks)
for z in range(num_chunks):
    np.save("/mnt/data"+str(z),np.transpose(convert_onehot2D(train_dna_seqs[z*chunk_size:(z+1)*chunk_size]), axes=(0,2,1)))
    np.save("/mnt/labels"+str(z),train_pi_labels_onehot[z*chunk_size:(z+1)*chunk_size])



