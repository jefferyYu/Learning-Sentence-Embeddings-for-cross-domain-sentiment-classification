# PyTC Functions V3.30, last updated on 2013-03-13.

import os, re, random, math
import numpy as np

def save_samps(samp_dict_list, samp_class_list, fname, feat_num = 0):
    length = len(samp_class_list)
    fout = open(fname, 'w')
    for k in range(length):
        samp_dict = samp_dict_list[k]
        samp_class = samp_class_list[k]
        fout.write(str(samp_class) + '\t')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()
    
    
def save_class_set(class_set, fname):
    open(fname, 'w').writelines([x + '\n' for x in class_set])
    
def save_term_set(term_set, fname):
    open(fname, 'w').writelines([x + '\n' for x in term_set])
    #open(fname, 'w').writelines([str(term_set.index(x)+1) + '\t\t' + x + '\n' for x in term_set])    

def get_term_set(doc_terms_list):
    term_set = set()
    for doc_terms in doc_terms_list:
        term_set.update(doc_terms)
    return sorted(list(term_set))

def get_doc_bis_list(doc_str_list):
    unis_list = [x.split() for x in doc_str_list]
    doc_bis_list = []
    for k in range(len(doc_str_list)):
        unis = unis_list[k]
        if len(unis) <= 1:
            doc_bis_list.append([])
            continue
        unis_shift = unis[1:] + [unis[0]]
        bis = [unis[j]+'<w-w>'+unis_shift[j] for j in range(len(unis))][0:-1]
        doc_bis_list.append(bis)
    return doc_bis_list

def get_joint_sets(doc_terms_list1, doc_terms_list2):
	joint_list = []
	for k in range(len(doc_terms_list1)):
		doc_terms1 = doc_terms_list1[k]
		doc_terms2 = doc_terms_list2[k]
		joint_list.append(doc_terms1 + doc_terms2)
	return joint_list
    
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9()_,!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()    

def get_class_set(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    return class_set

def get_doc_terms_list(doc_str_list):
    term_dict ={}
    for x in doc_str_list:
	word_list = clean_str(x).split()
	for word in word_list:
	    if not term_dict.has_key(word):
		term_dict[word] = 1
	    else:
		term_dict[word] += 1
    return [ clean_str(x).split() for x in doc_str_list], term_dict

def get_doc_terms_list_ori(doc_str_list):
    for x in doc_str_list:
	word_list = clean_str(x).split()
    return [ clean_str(x).split() for x in doc_str_list]

def build_samps(term_dict, class_dict, doc_terms_list, doc_class_list, term_weight, idf_term = None):
    '''
    New functions for building samples to sparse format from term list, 2010-3-23
    term_dict, for example, term1: 1; term2:2; term3:3, ...
    class_dict, for example, negative:1; postive:2; unlabel:0 
    '''
    samp_dict_list = []
    samp_class_list = []
    for k in range(len(doc_class_list)):
        doc_class = doc_class_list[k]
        samp_class = class_dict[doc_class]
        samp_class_list.append(samp_class)
        doc_terms = doc_terms_list[k]
        samp_dict = {}
	for term in doc_terms:
	    if term_dict.has_key(term):
		term_id = term_dict[term]
		if term_weight == 'BOOL':
		    samp_dict[term_id] = 1
		elif term_weight == 'TF':
		    if samp_dict.has_key(term_id):
			samp_dict[term_id] += 1
		    else:
			samp_dict[term_id] = 1
		elif term_weight == 'TFIDF':
		    if samp_dict.has_key(term_id):
			samp_dict[term_id] += idf_term[term]
		    else:
			samp_dict[term_id] = idf_term[term]
        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

def stat_df_class(class_set, doc_class_list):
    '''
    df_class is a list
    '''
    df_class = [doc_class_list.count(x) for x in class_set]
    return df_class

def load_df_term_class(fname):
    df_term_class = {};
    for line in open(fname, 'r'):
        term = line.strip().split()[0]
        df = [int(x) for x in line.strip().split()[1:]]
        df_term_class[term] = df
    return df_term_class

def stat_idf_term(doc_num, df_term):
    '''
    idf_term is a dict
    '''
    idf_term = {}.fromkeys(df_term.keys())
    for term in idf_term:
	    idf_term[term] = math.log(float(doc_num/df_term[term]))
    return idf_term


def supervised_feature_selection(df_class, df_term_class, fs_method = 'IG', fs_num = 0, fs_class = -1):
    if fs_method == 'MI':
	term_set_fs, term_score_list = feature_selection_mi(df_class, df_term_class, fs_num, fs_class)
    elif fs_method == 'WLLR':
	term_set_fs, term_score_list = feature_selection_wllr(df_class, df_term_class, fs_num, fs_class)
    return term_set_fs, term_score_list

def feature_selection_mi(df_class, df_term_class, fs_num = 0, fs_class = -1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:	
	df_list = df_term_class[term]
	class_set_size = len(df_list)
	term_set_size = len(df_term_class)
	N = sum(df_class)
	score_list = []
	for class_id in range(class_set_size):
	    A = df_list[class_id]
	    B = sum(df_list) - A
	    C = df_class[class_id] - A
	    D = N-A-C-B
	    p_c_t = (A+1.0)/(A+B+class_set_size) # A/(A+B) with add-one estimator
	    p_c = float(A+C)/N
	    score = math.log(p_c_t/p_c) # log(A*N/((A+C)(A+B)))
	    #p_t_c = (A+1.0)/(A+C+term_set_size) # A/(A+C) with add-one estimator, poor performance!
	    #p_t = float(A+B)/N #(A+B)/N
	    #score = math.log(p_t_c/p_t) # log(A*N/((A+C)(A+B)))
	    score_list.append(score)
	if fs_class == -1:
	    term_score = max(score_list) # max score
	else:
	    term_score = score_list[fs_class]
	term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key = lambda x:-x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
	    term_set_fs = term_set_rank
    else:
	    term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_list
	
def feature_selection_wllr(df_class, df_term_class, fs_num = 0, fs_class = -1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    term_slist_dict = {}.fromkeys(term_set)
    for term in term_set:	
	df_list = df_term_class[term]
	class_set_size = len(df_list)
	doc_set_size = len(df_class)
	N = sum(df_class)
	term_set_size=len(df_term_class)
	score_list = []
	for class_id in range(class_set_size):
	    A = df_list[class_id]
	    B = sum(df_list) - A
	    C = df_class[class_id] - A
	    D = N-A-C-B
	    p_t_c = (A+1E-6)/(A+C+1E-6*term_set_size) # A/(A+C) with add-one estimator
	    p_t_not_c=(B+1E-6)/(B+D+1E-6*term_set_size) # B/(B+D)
	    score = p_t_c * math.log(p_t_c/p_t_not_c)
	    score_list.append(score)
	if fs_class == -1:
	    term_score = max(score_list) # max score
	else:
	    term_score = score_list[fs_class]
	term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key = lambda x:-x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
	term_set_fs = term_set_rank
    else:
	term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_list

