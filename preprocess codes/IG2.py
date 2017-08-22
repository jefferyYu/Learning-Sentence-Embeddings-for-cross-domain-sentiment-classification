import os, sys, math, random
import numpy as np
import pytc
import nltk

def read_fbyline(fname_list):
        cls_list = []
	doc_list = []
        fin = open(fname_list,'r')
        for str_line in fin.readlines():
		samp_dict = {}
                doc_str = int(str_line.split(' ')[0])
                cls_list.append(doc_str)
		key_value_list = str_line.split(' ')[1:]
		for key_value in key_value_list:
			if key_value != '\n':
				key = int(key_value.split(':')[0])
			        value = float(key_value.split(':')[1])
			        samp_dict[key] = value
		doc_list.append(samp_dict)
	
        return doc_list, cls_list

def read_term_rev(fname_term_set):
	term_dict = {}
	fin = open(fname_term_set, 'r')
	for line in fin.readlines():
		key = int(line.split(' ')[1])
		value = line.split(' ')[0]
		term_dict[key] = value
	return term_dict

def read_term(fname_term_set):
	term_dict = {}
	fin = open(fname_term_set, 'r')
	for line in fin.readlines():
		key = int(line.split(' ')[1])
		value = line.split(' ')[0]
		term_dict[value] = key
	return term_dict
	
def stat_df_term_class(term_set, class_set, doc_terms_list, doc_class_list):
	'''
	df_term_class is a dict-list
	
	'''
	class_id_dict = dict(zip(class_set, range(len(class_set))))
	df_term_class = {}
	for term in term_set:
	    df_term_class[term] = [0]*len(class_set)
	for k in range(len(doc_class_list)):
		class_label = doc_class_list[k]
		class_id = class_id_dict[class_label]
		samp_dict = doc_terms_list[k]
		for term in samp_dict:
			if df_term_class.has_key(term):
				df_term_class[term][class_id] += 1
	return df_term_class

def save_term_score(term_score_list, fname, term_dict_rev, pivot_term):
    fout = open(fname, 'w')
    for term_score in term_score_list:
	    term = term_dict_rev[term_score[0]]
	    if term in pivot_term:
		    fout.write(term + '\t' + str(term_score[1]) + '\n')
    fout.close()
    
def save_term_score2(term_score_list, fname, term_dict_rev):
    fout = open(fname, 'w')
    for term_score in term_score_list:
	    term = term_dict_rev[term_score[0]]
	    fout.write(term + '\t' + str(term_score[1]) + '\n')
    fout.close()
    
def read_text_f(fname_input):
	fin_list = open(fname_input, 'r').readlines()
	class_list = []
	doc_list = []
	for i in xrange(len(fin_list)):
	    line = fin_list[i]
	    polarity = line.strip().split(' ')[0]
	    sentence = line.strip()[2:]
	    class_list.append(polarity)
	    doc_list.append(sentence)
	    #if polarity == 'positive':
		#fout.write('1' + ' ' + sentence + '\n')
	    #elif polarity == 'negative':
		#fout.write('0' + ' ' + sentence + '\n')	
	return doc_list, class_list

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

def unis_filter(fname_neg_score, fname_pos_score, adverse_list, postag_list, fs_percent):
    neg_postag_list = pos_list_filter(fname_neg_score, adverse_list, postag_list)
    pos_postag_list = pos_list_filter(fname_pos_score, adverse_list, postag_list)
    dict_len= int(min(len(neg_postag_list), len(pos_postag_list)) * fs_percent)
    return neg_postag_list[:dict_len], pos_postag_list[:dict_len]

def pos_list_filter(fname_socre, adverse_list, pos_list):
    filter_pos_list = []
    pol_list =  [ (line.strip().split())[0] for line in open(fname_socre, 'r').readlines() ] 
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')    
    stopword_short = ["'m", "'s", "'re", "'ve", "e", "d"]
    for k in range(len(pol_list)):
        term = pol_list[k]
	if '_' not in term:
		continue;
        term_list = term.split('_')
        if term_list[1] in pos_list and term_list[0] not in adverse_list and term_list[0] not in stopwords and term_list[0] not in stopword_short:
            filter_pos_list.append(term_list[0])
    return filter_pos_list

if __name__ == '__main__':
    
	jj_pos_list = ['jj' ,'jjs' ,'jjr' ,'rb' , 'rbs' , 'rbr'] 
	v_pos_list = ['vb' ,'vbz' , 'vbd' , 'vbn' , 'vbg' , 'vbp'] 	
	
	adverse_list = ['not', 'no', 'without', 'never', 'n\'t', 'don\'t', 'hardly']
	
	src_list = ['stsa.binary', 'rt-polarity', 'custrev', 'laptop', 'restaurant'] #'stsa.binary', 'rt-polarity', 'custrev', 'laptop', 'restaurant'
	tgt_list = ['stsa.binary', 'rt-polarity', 'custrev', 'laptop', 'restaurant'] #'laptop', 'restaurant'
        for src in src_list:
		for tgt in tgt_list:
			if src == tgt:
				continue;
			else:
				if src == 'stsa.binary' and tgt == 'rt-polarity':
				    continue;
				if tgt == 'stsa.binary' and src == 'rt-polarity':
				    continue;
				else:				
					print 'Selecting features...'    
					fs_method = 'WLLR'
					term_weight = 'BOOL'
					fs_num = 0
					fname_feat_score_st1 = 'termset.Neg.rank_pos'
					fname_feat_score_st2 = 'termset.Pos.rank_pos'
					
					pivot_dir = 'pivotlist'
					if not os.path.exists(pivot_dir):
					    os.makedirs(pivot_dir)					
	
					fname_filter_neg = pivot_dir + os.sep + src[:2] + tgt[:2] + '_termset.Negfilter'
					fname_filter_pos = pivot_dir + os.sep + src[:2] + tgt[:2] + '_termset.Posfilter'
					
					fs_percent = 0.25
					
					doc_str_class_list_train = []
					source_input = 'tag_data'+os.sep+src + '.pos'
					doc_str_list_per, doc_class_list_per = read_text_f(source_input)
					doc_str_class_list_per = zip(doc_str_list_per, doc_class_list_per)
					doc_str_class_list_train.extend(doc_str_class_list_per)
					
					doc_str_class_list_target = []
					target_input = 'tag_data'+os.sep+tgt + '.pos'
					doc_str_list_per_target, doc_class_list_per_target = read_text_f(target_input)
					doc_str_class_list_per_target = zip(doc_str_list_per_target, doc_class_list_per_target)
					doc_str_class_list_target.extend(doc_str_class_list_per_target)	
					
					doc_str_list_train = []
					doc_class_list_train = []
					for doc_str_class in doc_str_class_list_train:
					    #print(doc_str_class[0]) 
					    #print(doc_str_class[1])
					    doc_str_list_train.append(doc_str_class[0])
					    doc_class_list_train.append(doc_str_class[1])  	
					    
					doc_str_list_test = []
					doc_class_list_test = []
					for doc_str_class in doc_str_class_list_target:
					    doc_str_list_test.append(doc_str_class[0])
					    doc_class_list_test.append(doc_str_class[1])	
					    
					print 'Extracting features...'
					# unigrams
					doc_unis_list_train,train_dict = pytc.get_doc_terms_list(doc_str_list_train)
					doc_unis_list_test,test_dict = pytc.get_doc_terms_list(doc_str_list_test)	
					class_set = pytc.get_class_set(doc_class_list_train)
					class_dict = dict(zip(class_set, range(1,1+len(class_set))))
				
					train_term_set = train_dict.keys()
					test_term_set = test_dict.keys()
					term_set = []
					term_set.extend(train_term_set)
					term_set.extend(test_term_set)
					term_set = list(set(term_set))
					term_dict = dict(zip(term_set, range(1, 1+len(term_set))))	
					samp_list_train, class_list_train = pytc.build_samps(term_dict, class_dict, doc_unis_list_train, doc_class_list_train, term_weight)
					samp_list_test, class_list_test = pytc.build_samps(term_dict, class_dict, doc_unis_list_test, doc_class_list_test, term_weight)  		
					term_dict_rev = dict(zip(range(1, 1+len(term_set)), term_set))
					train_term = []
					for term in train_dict.keys():
						if train_dict[term] > 2:
							train_term.append(term)
					print len(train_term)
					test_term = []
					for term in test_dict.keys():
						if test_dict[term] > 2:
							test_term.append(term)
					print len(test_term)
					pivot_term = list(set(train_term).intersection(set(test_term)))
					print len(pivot_term)
					class_set_st = [1,2]
					df_class_st = pytc.stat_df_class(class_set_st, class_list_train)
					df_term_class_st = stat_df_term_class(term_dict_rev, class_set_st, samp_list_train, class_list_train)
					neg_term_set_st, neg_term_score_dict_st = pytc.supervised_feature_selection(df_class_st, df_term_class_st, fs_method, fs_num, fs_class = 0)
					pos_term_set_st, pos_term_score_dict_st = pytc.supervised_feature_selection(df_class_st, df_term_class_st, fs_method, fs_num, fs_class = 1)
					save_term_score(neg_term_score_dict_st, fname_feat_score_st1, term_dict_rev, pivot_term)    
					save_term_score(pos_term_score_dict_st, fname_feat_score_st2, term_dict_rev, pivot_term)   
					
					jj_filter_list_neg, jj_filter_list_pos = unis_filter(fname_feat_score_st1, fname_feat_score_st2, adverse_list, jj_pos_list, fs_percent)
					v_filter_list_neg, v_filter_list_pos = unis_filter(fname_feat_score_st1, fname_feat_score_st2, adverse_list, v_pos_list, fs_percent)
					filter_list_neg = jj_filter_list_neg + v_filter_list_neg
					filter_list_pos = jj_filter_list_pos + v_filter_list_pos
					
					fout = open(fname_filter_neg, 'w')
					k=0
					while k != len(filter_list_neg):
					    fout.writelines(str(filter_list_neg[k])+'\n')
					    k=k+1
					fout.close()	
					
					fout = open(fname_filter_pos, 'w')
					k=0
					while k != len(filter_list_pos):
					    fout.writelines(str(filter_list_pos[k])+'\n')
					    k=k+1
					fout.close()	
