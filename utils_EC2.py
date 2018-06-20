# 2016-12-30

import numpy as np
import json
import random
import re
import string
import itertools
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import time
import pickle
from pubmed_lookup import PubMedLookup, Publication

def get_json_plasmids(filename):
    start = time.time()
    json_file = open(filename).read()
    json_tree = json.loads(json_file)
    end = time.time()
    print("Parsing JSON tree took", end-start, "seconds")
    print("")
    return json_tree['plasmids']


def count_plasmids_per_pmid(json_plasmids):
    start = time.time()
    
    pmid_count_dict = {}
    doi_count_dict = {}
    pmid_to_doi = {}
    for p in json_plasmids:
        doi = p['article']['doi']
        pmid = p['article']['pubmed_id']
        
        temp_convert = {pmid: doi}
        pmid_to_doi.update(temp_convert)
    
        if pmid in pmid_count_dict:
            temp_dict = {pmid: pmid_count_dict[pmid]+1}
            pmid_count_dict.update(temp_dict)
        else:
            temp_dict = {pmid: 1}
            pmid_count_dict.update(temp_dict)
            
            
        if doi in doi_count_dict:
            temp_dict = {doi: doi_count_dict[doi]+1}
            doi_count_dict.update(temp_dict)
        else:
            temp_dict = {doi: 1}
            doi_count_dict.update(temp_dict)
            

    end = time.time()
    print("Counting plasmids per PMID took", end-start, "seconds")
    print("")
    
    return pmid_count_dict, doi_count_dict, pmid_to_doi

def count_plasmids_per_year(pmid_counts, doi_counts, pmid_to_doi):
    start = time.time()
    year_count_dict = {}
    
    with open('PMC-ids.csv') as f:
        lines = f.readlines()
    del lines[0]
    
    PMID_year_dict = {}
    DOI_year_dict = {}
    for p in lines:
        k = p.split(',')
        temp_dict = {k[9]: k[3]}
        temp_dict2 = {k[7]: k[3]}
        PMID_year_dict.update(temp_dict)
        DOI_year_dict.update(temp_dict2)
        
    print("Length of pmid_counts =", len(pmid_counts))
    for i in pmid_counts:
        if str(i) in PMID_year_dict:                
            year = int(PMID_year_dict[str(i)])
        elif str(pmid_to_doi[i]) in DOI_year_dict:
            year = int(DOI_year_dict[str(pmid_to_doi[i])])
        else:
            email = ''
            url = 'http://www.ncbi.nlm.nih.gov/pubmed/' + str(i)
            lookup = PubMedLookup(url, email)
            publication = Publication(lookup)    # Use 'resolve_doi=False' to keep DOI URL
            year = publication.year
        
        if year in year_count_dict:
            temp_dict = {year: year_count_dict[year] + pmid_counts[i]}
            year_count_dict.update(temp_dict)
        else:
            temp_dict = {year: pmid_counts[i]}
            year_count_dict.update(temp_dict)
        
    end = time.time()
    
    print("Fetching publication year for all PMIDs took", end-start, "seconds")
    print("")
    
    return year_count_dict

def get_num_plasmids_per_pi(json_plasmids, min_num_plasmids_cutoff, max_seq_length):
    start = time.time()
    pi_plasmid_dict = {}
    no_pi = 0
    for p in json_plasmids:
        seqlen = 0
        if len(p['sequences']['public_addgene_full_sequences']) > 0:
            for t in p['sequences']['public_addgene_full_sequences']:
                seqlen += len(convert_seq_to_atgcn(t))
        elif len(p['sequences']['public_user_full_sequences']) > 0:
            for u in p['sequences']['public_user_full_sequences']:
                seqlen += len(convert_seq_to_atgcn(u))
        else:
            for v in p['sequences']['public_addgene_partial_sequences']:
                seqlen += len(convert_seq_to_atgcn(v))
            for w in p['sequences']['public_user_partial_sequences']:
                seqlen += len(convert_seq_to_atgcn(w))

        if seqlen > 0:
            if len(p["pi"]) > 0: pi_name = ' & '.join(p["pi"])
            else: pi_name = "No PI"
            if pi_name in pi_plasmid_dict:
                temp_dict = {pi_name: pi_plasmid_dict[pi_name]+1}
                pi_plasmid_dict.update(temp_dict)
            else:
                temp_dict = {pi_name: 1}
                pi_plasmid_dict.update(temp_dict)
                
                
    dict_copy = dict(pi_plasmid_dict)
    for pi in pi_plasmid_dict:
        if pi_plasmid_dict[pi] < min_num_plasmids_cutoff: del dict_copy[pi]
    print("Number of PIs with at least", min_num_plasmids_cutoff, "plasmids =", len(dict_copy))
    print("Number of remaining plasmids =", sum(dict_copy.values()))
    end = time.time()
    print("Reducing PI list took", end-start, "seconds")
    print("")
    return dict_copy


def parseTree(obj):  
    if isinstance(obj,int) or isinstance(obj,type(None)): 
        pass
    elif isinstance(obj,str):
        entry = ''.join(re.sub(r'\W+', '', obj))
        if len(obj)>0 and len(obj)<60:
            leaf_array.append(entry.lower())
    elif isinstance(obj,list):
        for child in obj:
            parseTree(child)   
    else:
        for child in obj:
            parseTree(obj[child])

def convert_seq_to_atgcn(seq):
    char_list = list(string.printable)
    for i in ['A','T','G','C','a','t','g','c']:
        char_list.remove(i)
    for ch in char_list:
        if ch in seq:
            seq=seq.replace(ch,"N")
    return seq.upper()

            
def get_seqs_annotations(json_plasmids, pi_plasmid_dict, filter_length, max_seq_length):
    start = time.time()
    num_remaining_plasmids = sum(pi_plasmid_dict.values())
    remaining_pis = pi_plasmid_dict.keys()
    pis = [''] * num_remaining_plasmids
    plasmid_names = [''] * num_remaining_plasmids
    seqs = [''] * num_remaining_plasmids
    annotations = [[] for i in range(num_remaining_plasmids)]
    count = 0
    global leaf_array
    for p in json_plasmids:
        seqlen = 0
        if len(p['sequences']['public_addgene_full_sequences']) > 0:
            for t in p['sequences']['public_addgene_full_sequences']:
                seqlen += len(convert_seq_to_atgcn(t))
        elif len(p['sequences']['public_user_full_sequences']) > 0:
            for u in p['sequences']['public_user_full_sequences']:
                seqlen += len(convert_seq_to_atgcn(u))
        else:
            for v in p['sequences']['public_addgene_partial_sequences']:
                seqlen += len(convert_seq_to_atgcn(v))
            for w in p['sequences']['public_user_partial_sequences']:
                seqlen += len(convert_seq_to_atgcn(w))
        
      
        if seqlen > 0:
            if len(p["pi"]) > 0:
                if ' & '.join(p["pi"]) in remaining_pis:
                    # If PI in list above submission threshold, concatenate seqs, add annotations
                    pis[count] = (' & '.join(p["pi"]))
                    
                    if len(p['sequences']['public_addgene_full_sequences']) > 0:
                        for t in p['sequences']['public_addgene_full_sequences']:
                            seqs[count] += ((convert_seq_to_atgcn(t) + 'N'*filter_length))#.encode())
                    elif len(p['sequences']['public_user_full_sequences']) > 0:
                        for u in p['sequences']['public_user_full_sequences']:
                            seqs[count] += ((convert_seq_to_atgcn(u) + 'N'*filter_length))#.encode())
                    else:
                        for v in p['sequences']['public_addgene_partial_sequences']:
                            seqs[count] += ((convert_seq_to_atgcn(v) + 'N'*filter_length))#.encode())
                        for w in p['sequences']['public_user_partial_sequences']:
                            seqs[count] += ((convert_seq_to_atgcn(w) + 'N'*filter_length))#.encode())
                            
                    if len(seqs[count]) > max_seq_length:
                        seqs[count] = seqs[count][0:max_seq_length]
                            
                    if len(p["name"]) > 0:
                        plasmid_names[count] = p["name"]
                    else:
                        plasmid_names[count] = "None"
                    
                    leaf_array = []
                    parseTree(p['bacterial_resistance'])
                    parseTree(p['cloning'])
                    if len(p['inserts']) > 0:
                        parseTree(p['inserts'][0]['alt_names'])
                        parseTree(p['inserts'][0]['cloning'])
                        if len(p['inserts'][0]['entrez_gene']) > 0:
                            #parseTree(p['inserts'][0]['entrez_gene'][0]['aliases']) # in JSON file, "list" of aliases string, not list
                            parseTree(p['inserts'][0]['entrez_gene'][0]['gene'])
                    annotations[count] = leaf_array
                    count += 1
    end = time.time()
    print("Getting DNA seqs and annotations for remaining PIs took", end-start, "seconds")
    print("")
    return [pis, seqs, annotations, plasmid_names]

def permute_order(pis, seqs, annotations):
    permute = np.random.permutation(len(pis))
    pis_permute = [''] * len(pis)
    seqs_permute = [''] * len(seqs)
    annotations_permute = [''] * len(annotations)
    count = 0
    for i in permute:
        pis_permute[count] = pis[i]
        seqs_permute[count] = seqs[i]
        annotations_permute[count] = annotations[i]
        count += 1
    return pis_permute, seqs_permute, annotations_permute

def convert_pi_labels_onehot(pi_labels):
    sorted_pis = sorted(pi_labels)
    set_pis = list(sorted_pis for sorted_pis,_ in itertools.groupby(sorted_pis))
    labels_onehot = np.zeros((len(pi_labels),len(set_pis)))
    for i in range(len(pi_labels)):
        labels_onehot[i][set_pis.index(pi_labels[i])] = 1
    return [set_pis, labels_onehot]

def pad_dna(seqs,maxlen):
    start = time.time()
    padded_seqs = [''] * len(seqs)
    for i in seqs:
        if len(i) > maxlen:
            i = i[:maxlen]
            maxlen = len(i)
    #print("Maximum DNA sequence length =", maxlen, "nt")
    for j in range(len(seqs)):
        if len(seqs[j]) > maxlen:
            seq = seqs[j][0:maxlen]
        else:
            seq = seqs[j]
        padded_seqs[j] = seq + "N" * (maxlen - len(seq))
    end = time.time()
    #print("N-padding DNA took", end-start, "seconds")
    #print("")
    return padded_seqs

def append_rc(seqs,filter_length):
    start = time.time()
    full_seqs = [''] * len(seqs)
    rc_dict = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
    for j in range(len(seqs)):
        fwd_seq = seqs[j]
        complement_seq = ''
        for n in fwd_seq:
            complement_seq += rc_dict[n]
        full_seqs[j] = fwd_seq + 'N'*filter_length + complement_seq[::-1] #[::-1] reverses string 
    end = time.time()
    #print("Appending reverse complement took", end-start, "seconds")
    #print("")
    return full_seqs


def convert_annotations(annotations):
    start = time.time()
    sorted_annotations_list = []
    for i in annotations:
        for j in i:
            if j not in sorted_annotations_list:
                sorted_annotations_list.append(j)
    sorted_annotations_list.sort()
    print("Total number of unique annotations across remaining plasmids =", len(sorted_annotations_list))
    annotations_bow = np.zeros((len(annotations),len(sorted_annotations_list)))
    for k in range(len(annotations)):
        for a in annotations[k]:
            annotations_bow[k][sorted_annotations_list.index(a)] += 1
    end = time.time()
    print("Bag of words conversion took", end-start, "seconds")
    print("")
    return [sorted_annotations_list, annotations_bow]

def feature_selection(pi_labels_onehot, annotation_labels_bow, sorted_annotations, normparam):
    start = time.time()
    y_category = np.zeros((len(pi_labels_onehot)))
    for i in range(len(pi_labels_onehot)):
        y_category[i] = np.argmax(pi_labels_onehot[i])
    lsvc = LinearSVC(C=normparam, penalty="l1", dual=False, class_weight='balanced',max_iter=50).fit(annotation_labels_bow, y_category)
    model = SelectFromModel(lsvc, prefit=True)
    reduced_annotation_labels_bow = model.transform(annotation_labels_bow)
    sorted_reduced_annotations = model.transform((np.array(sorted_annotations)).reshape(1,-1))
    print("Reduced annotations shape =", reduced_annotation_labels_bow.shape)
    end = time.time()
    print("Feature selection took", end-start, "seconds")
    print("")
    return [sorted_reduced_annotations, reduced_annotation_labels_bow]
    
def separate_train_val_test(params, sorted_pi_list, sorted_annotations, sorted_reduced_annotations, pi_labels, pi_labels_onehot, dna_seqs, annotation_labels, annotation_labels_bow, reduced_annotation_labels_bow, timestamp):
    
    min_num_plasmids_cutoff = params[0]
    val_plasmids_per_pi = params[3]
    test_plasmids_per_pi = params[4]
    
    assert val_plasmids_per_pi + test_plasmids_per_pi < min_num_plasmids_cutoff
    pi_val_count = [0] * len(sorted_pi_list)
    pi_test_count = [0] * len(sorted_pi_list)
    num_training_rows = len(pi_labels) - val_plasmids_per_pi * len(sorted_pi_list) - test_plasmids_per_pi * len(sorted_pi_list)
    
    train_pi_labels = [''] * num_training_rows
    train_pi_labels_onehot = np.zeros((num_training_rows,len(pi_labels_onehot[0])))
    train_dna_seqs = [''] * num_training_rows
    train_annotation_labels = [[] for i in range(num_training_rows)]
    #train_annotation_labels_bow = np.zeros((num_training_rows,len(annotation_labels_bow[0])))
    train_reduced_annotation_labels_bow = np.zeros((num_training_rows,len(reduced_annotation_labels_bow[0])))
    
    val_pi_labels = [''] * (val_plasmids_per_pi * len(sorted_pi_list))
    val_pi_labels_onehot = np.zeros((val_plasmids_per_pi * len(sorted_pi_list),len(pi_labels_onehot[0])))
    val_dna_seqs = [''] * (val_plasmids_per_pi * len(sorted_pi_list))
    val_annotation_labels = [[] for i in range(val_plasmids_per_pi * len(sorted_pi_list))]
    #val_annotation_labels_bow = np.zeros((val_plasmids_per_pi * len(sorted_pi_list),len(annotation_labels_bow[0])))
    val_reduced_annotation_labels_bow = np.zeros((val_plasmids_per_pi * len(sorted_pi_list),len(reduced_annotation_labels_bow[0])))
    
    test_pi_labels = [''] * (test_plasmids_per_pi * len(sorted_pi_list))
    test_pi_labels_onehot = np.zeros((test_plasmids_per_pi * len(sorted_pi_list),len(pi_labels_onehot[0])))
    test_dna_seqs = [''] * (test_plasmids_per_pi * len(sorted_pi_list))
    test_annotation_labels = [[] for i in range(test_plasmids_per_pi * len(sorted_pi_list))]
    #test_annotation_labels_bow = np.zeros((test_plasmids_per_pi * len(sorted_pi_list),len(annotation_labels_bow[0])))
    test_reduced_annotation_labels_bow = np.zeros((test_plasmids_per_pi * len(sorted_pi_list),len(reduced_annotation_labels_bow[0])))
    
    train_row = 0
    val_row = 0
    test_row = 0
    for i in range(len(pi_labels)):
        if pi_val_count[sorted_pi_list.index(pi_labels[i])] < val_plasmids_per_pi:
            val_pi_labels[val_row] = pi_labels[i]
            val_pi_labels_onehot[val_row] = pi_labels_onehot[i]
            val_dna_seqs[val_row] = dna_seqs[i]
            val_annotation_labels[val_row] = annotation_labels[i]
            #val_annotation_labels_bow[val_row] = annotation_labels_bow[i]
            val_reduced_annotation_labels_bow[val_row] = reduced_annotation_labels_bow[i]
            pi_val_count[sorted_pi_list.index(pi_labels[i])] += 1
            val_row += 1
        elif pi_test_count[sorted_pi_list.index(pi_labels[i])] < test_plasmids_per_pi:
            test_pi_labels[test_row] = pi_labels[i]
            test_pi_labels_onehot[test_row] = pi_labels_onehot[i]
            test_dna_seqs[test_row] = dna_seqs[i]
            test_annotation_labels[test_row] = annotation_labels[i]
            #test_annotation_labels_bow[test_row] = annotation_labels_bow[i]
            test_reduced_annotation_labels_bow[test_row] = reduced_annotation_labels_bow[i]
            pi_test_count[sorted_pi_list.index(pi_labels[i])] += 1
            test_row += 1
        else:
            train_pi_labels[train_row] = pi_labels[i]
            train_pi_labels_onehot[train_row] = pi_labels_onehot[i]
            train_dna_seqs[train_row] = dna_seqs[i]
            train_annotation_labels[train_row] = annotation_labels[i]
            #train_annotation_labels_bow[train_row] = annotation_labels_bow[i]
            train_reduced_annotation_labels_bow[train_row] = reduced_annotation_labels_bow[i]
            train_row += 1
    
    cl_weight = {}
    for i in range(train_pi_labels_onehot.shape[1]):
        cl_weight[i] = 0
    for x in range(train_pi_labels_onehot.shape[0]):
        cl_weight[np.argmax(train_pi_labels_onehot[x,:])] += 1
    sumval = sum(cl_weight.values())
    for y in cl_weight.keys():
        cl_weight[y] = len(cl_weight)*float(cl_weight[y])/float(sumval)
    
    t = '/mnt/' + timestamp
    #t = timestamp
    np.save(t + '_params.out', params)
    pickle.dump(sorted_pi_list, open(t + "_sorted_pi_list.out", "wb"))
    pickle.dump(sorted_annotations, open(t + "_sorted_annotations.out", "wb"))
    pickle.dump(sorted_reduced_annotations, open(t + "_sorted_reduced_annotations.out", "wb"))
    pickle.dump(train_pi_labels, open(t + "_train_pi_labels.out", "wb"))
    np.save(t + '_train_pi_labels_onehot.out', train_pi_labels_onehot)
    pickle.dump(train_dna_seqs, open(t + "_train_dna_seqs.out", "wb"))
    pickle.dump(train_annotation_labels, open(t + "_train_annotation_labels.out", "wb"))
    #np.save(t + '_train_annotation_labels_bow.out', train_annotation_labels_bow)
    np.save(t + '_train_reduced_annotation_labels_bow.out', train_reduced_annotation_labels_bow)
    pickle.dump(cl_weight, open(t + "_class_weight.out", "wb"))
    pickle.dump(val_pi_labels, open(t + "_val_pi_labels.out", "wb"))
    np.save(t + '_val_pi_labels_onehot.out', val_pi_labels_onehot)
    pickle.dump(val_dna_seqs, open(t + "_val_dna_seqs.out", "wb"))
    pickle.dump(val_annotation_labels, open(t + "_val_annotation_labels.out", "wb"))
    #np.save(t + '_val_annotation_labels_bow.out', val_annotation_labels_bow)
    np.save(t + '_val_reduced_annotation_labels_bow.out', val_reduced_annotation_labels_bow)
    pickle.dump(test_pi_labels, open(t + "_test_pi_labels.out", "wb"))
    np.save(t + '_test_pi_labels_onehot.out', test_pi_labels_onehot)
    pickle.dump(test_dna_seqs, open(t + "_test_dna_seqs.out", "wb"))
    pickle.dump(test_annotation_labels, open(t + "_test_annotation_labels.out", "wb"))
    #np.save(t + '_test_annotation_labels_bow.out', test_annotation_labels_bow)
    np.save(t + '_test_reduced_annotation_labels_bow.out', test_reduced_annotation_labels_bow)

def convert_onehot2D(list_of_seqs):
    list_of_onehot2D_seqs = np.zeros((len(list_of_seqs),4,len(list_of_seqs[0])))
    nt_dict = {'A':[1,0,0,0],'T':[0,1,0,0],'G':[0,0,1,0],'C':[0,0,0,1], 'N':[0,0,0,0]}
    count = 0
    for seq in list_of_seqs:
        if len(seq) > 1:
            for letter in range(len(seq)):
                for i in range(4):
                    list_of_onehot2D_seqs[count][i][letter] = (nt_dict[seq[letter]])[i]
        count += 1
    return list_of_onehot2D_seqs
        
        
        
        
        
        
        
        
        
        
        
