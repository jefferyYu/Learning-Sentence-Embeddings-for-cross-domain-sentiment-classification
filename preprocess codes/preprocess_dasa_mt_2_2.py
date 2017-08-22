import numpy as np
import h5py
import re
import sys
import operator
import argparse
import os
import random

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def line_to_words(line, dataset):
  if dataset == 'SST1' or dataset == 'SST2':
    clean_line = clean_str_sst(line.strip())
  else:
    clean_line = clean_str(line.strip())
  words = clean_line.split(' ')
  words = words[1:]

  return words

def get_vocab(file_list, dataset=''):
  max_sent_len = 0
  word_to_idx = {}
  # Starts at 2 for padding
  idx = 2

  for filename in file_list:
    f = open(filename, "r")
    for line in f:
        words = line_to_words(line, dataset)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = idx
                idx += 1

    f.close()

  return max_sent_len, word_to_idx

def load_data(dataset, train_name, test_name='', dev_name='', padding=4):
  """
  Load training data (dev/test optional).
  """
  f_names = [train_name]
  if not test_name == '': f_names.append(test_name)
  if not dev_name == '': f_names.append(dev_name)

  max_sent_len, word_to_idx = get_vocab(f_names, dataset)

  dev = []
  dev_label = []
  train = []
  train_label = []
  test = []
  test_label = []

  files = []
  data = []
  data_label = []

  f_train = open(train_name, 'r')
  files.append(f_train)
  data.append(train)
  data_label.append(train_label)
  if not test_name == '':
    f_test = open(test_name, 'r')
    files.append(f_test)
    data.append(test)
    data_label.append(test_label)
  if not dev_name == '':
    f_dev = open(dev_name, 'r')
    files.append(f_dev)
    data.append(dev)
    data_label.append(dev_label)

  for d, lbl, f in zip(data, data_label, files):
    for line in f:
      words = line_to_words(line, dataset)
      y = int(line[0]) + 1
      sent = [word_to_idx[word] for word in words]
      # end padding
      if len(sent) < max_sent_len + padding:
          sent.extend([1] * (max_sent_len + padding - len(sent)))
      # start padding
      sent = [1]*padding + sent

      d.append(sent)
      lbl.append(y)

  f_train.close()
  if not test_name == '':
    f_test.close()
  if not dev_name == '':
    f_dev.close()

  return word_to_idx, np.array(train, dtype=np.int32), np.array(train_label, dtype=np.int32), np.array(test, dtype=np.int32), np.array(test_label, dtype=np.int32), np.array(dev, dtype=np.int32), np.array(dev_label, dtype=np.int32)

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
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

def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

args = {}

def main(w2v_path):
  global args
  parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--train', help="custom train data", type=str, default="")
  parser.add_argument('--test', help="custom test data", type=str, default="")
  parser.add_argument('--dev', help="custom dev data", type=str, default="")
  parser.add_argument('--padding', help="padding around each sentence", type=int, default=4)
  parser.add_argument('--custom_name', help="name of custom output hdf5 file", type=str, default="custom")
  args = parser.parse_args()
  
  src_list = ['stsa.binary', 'rt-polarity', 'custrev', 'laptop', 'restaurant'] #'stsa.binary', 'rt-polarity', 'custrev', 'laptop', 'restaurant'
  tgt_list = ['stsa.binary', 'rt-polarity', 'custrev', 'laptop', 'restaurant'] #'stsa.binary', 'rt-polarity', 'custrev', 'laptop', 'restaurant'
  
  adverse_list = ['not', 'no', 'without', 'never', 'n\'t', 'don\'t', 'hardly']
  
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
			  FILE_PATHS = {
			                'RDASA_MT_4_' + src[:2] + tgt[:2]:('data'+os.sep + src + os.sep + src + '.txt', 'data'+os.sep + tgt + os.sep + tgt +'.txtdev', 'data'+os.sep + tgt + os.sep + tgt +'.txttest'),
			                }  
			  dataset = 'RDASA_MT_4_' + src[:2] + tgt[:2]
			
			  # Dataset name
			  train_path, dev_path, test_path = FILE_PATHS[dataset]
			
			  # Load data
			  word_to_idx, train, train_label, test, test_label, dev, dev_label = load_data(dataset, train_path, test_name=test_path, dev_name=dev_path, padding=args.padding)
			
			  # Write word mapping to text file.
			  with open(dataset + '_word_mapping.txt', 'w+') as embeddings_f:
			    embeddings_f.write("*PADDING* 1\n")
			    for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
			      embeddings_f.write("%s %d\n" % (word, idx))
			  
			  '''load pivot words'''
			  pivot_dir = 'pivotlist'
			  fname_Neg = pivot_dir + os.sep + src[:2] + tgt[:2] + '_termset.Negfilter'
			  fname_Pos = pivot_dir + os.sep + src[:2] + tgt[:2] + '_termset.Posfilter'
			  
			  fin_neg = open(fname_Neg, 'r').readlines()
			  fin_pos = open(fname_Pos, 'r').readlines()
			  neg_pivot_term = []
			  pos_pivot_term = []
			  for i in xrange(len(fin_neg)):
			      line = fin_neg[i]
			      term = line.strip()     
			      if word_to_idx.has_key(term):
				  neg_pivot_term.append(word_to_idx[term])
			  for j in xrange(len(fin_pos)):
			      line = fin_pos[j]
			      term = line.strip()  
			      if word_to_idx.has_key(term):
				  pos_pivot_term.append(word_to_idx[term])
				  
			  adverse_list_idx = []
			  for term in adverse_list:
			      if word_to_idx.has_key(term):
				  adverse_list_idx.append(word_to_idx[term])
			  
			  '''Training set from the source domain, new_train is for the samples without pivots, new_train_label is the label for our two auxiliary tasks'''
			  new_train_label = np.ones((len(train), 2), np.int32)
			  new_train = []
			  train_pivotnotin_index = []
			  for i in xrange(len(train)):
			      samp = train[i]
			      new_samp = []
			      neg_count = 0
			      pos_count = 0      

	                      for term_idx in range(len(samp)):
				  term = samp[term_idx]
				  if term in neg_pivot_term:
				      new_samp.append(1)
				      '''Detect if there is a negation word before the current word, if there is, reverse its sentiment label'''
				      term_pre1 = ''
				      term_pre2 = ''
				      term_pre3 = ''
				      #term_aft1= ''
				      #term_aft2 = ''
				      if term_idx ==1:
					  term_pre1 = samp[term_idx-1]
				      elif term_idx == 2:
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      elif term_idx >= 3:
					  term_pre3 = samp[term_idx-3]
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      if term_pre1 in adverse_list_idx or term_pre2 in adverse_list_idx or term_pre3 in adverse_list_idx:
					  new_train_label[i][1] = 2
					  pos_count += 1
				      else:
					  new_train_label[i][0] = 2
				          neg_count += 1
				  elif term in pos_pivot_term:
				      new_samp.append(1)
				      '''Detect if there is a negation word before the current word, if there is, reverse its sentiment label'''
				      term_pre1 = ''
				      term_pre2 = ''
				      term_pre3 = ''
				      #term_aft1= ''
				      #term_aft2 = ''
				      if term_idx ==1:
					  term_pre1 = samp[term_idx-1]
				      elif term_idx == 2:
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      elif term_idx >= 3:
					  term_pre3 = samp[term_idx-3]
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      if term_pre1 in adverse_list_idx or term_pre2 in adverse_list_idx or term_pre3 in adverse_list_idx:
					  new_train_label[i][0] = 2
					  neg_count += 1
				      else:
					  new_train_label[i][1] = 2
				          pos_count += 1
				  else:
				      new_samp.append(term)
			      if new_train_label[i][0] == new_train_label[i][1] == 1:
				  train_pivotnotin_index.append(i)
			      new_train.append(new_samp)
			  new_train = np.array(new_train, dtype=np.int32)
			  new_train_label = np.array(new_train_label, dtype=np.int32)
			  
			  '''Development set from the target domain, new_dev is for the samples without pivots, new_dev_label is the label for our two auxiliary tasks'''
			  new_dev_label = np.ones((len(dev), 2),np.int32)
			  dev_pivotin_index = []
			  new_dev = []
			  for i in xrange(len(dev)):
			      samp = dev[i]
			      new_samp = []
			      neg_count = 0
			      pos_count = 0
	                      for term_idx in range(len(samp)):
				  term = samp[term_idx]			      
				  if term in neg_pivot_term:
				      new_samp.append(1)
				      '''Detect if there is a negation word before the current word, if there is, reverse its sentiment label'''
				      term_pre1 = ''
				      term_pre2 = ''
				      term_pre3 = ''
				      #term_aft1= ''
				      #term_aft2 = ''
				      if term_idx ==1:
					  term_pre1 = samp[term_idx-1]
				      elif term_idx == 2:
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      elif term_idx >= 3:
					  term_pre3 = samp[term_idx-3]
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      if term_pre1 in adverse_list_idx or term_pre2 in adverse_list_idx or term_pre3 in adverse_list_idx:
					  new_dev_label[i][1] = 2
					  pos_count += 1
				      else:
					  new_dev_label[i][0] = 2
				          neg_count += 1				      
				  elif term in pos_pivot_term:
				      new_samp.append(1)
				      '''Detect if there is a negation word before the current word, if there is, reverse its sentiment label'''
				      term_pre1 = ''
				      term_pre2 = ''
				      term_pre3 = ''
				      #term_aft1= ''
				      #term_aft2 = ''
				      if term_idx ==1:
					  term_pre1 = samp[term_idx-1]
				      elif term_idx == 2:
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      elif term_idx >= 3:
					  term_pre3 = samp[term_idx-3]
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      if term_pre1 in adverse_list_idx or term_pre2 in adverse_list_idx or term_pre3 in adverse_list_idx:
					  new_dev_label[i][0] = 2
					  neg_count += 1
				      else:
					  new_dev_label[i][1] = 2
				          pos_count += 1
				  else:
				      new_samp.append(term)				      
			      if new_dev_label[i][0] !=1 or new_dev_label[i][1] != 1:
				  dev_pivotin_index.append(i)
			      new_dev.append(new_samp)
			  new_dev = np.array(new_dev, dtype=np.int32)
			  new_dev_label = np.array(new_dev_label, dtype=np.int32)  
			  
			  '''Test set from the target domain, new_test is for the samples without pivots, new_test_label is the label for our two auxiliary tasks'''
			  new_test_label = np.ones((len(test), 2),np.int32)
			  test_pivotin_index = []
			  new_test = []
			  for i in xrange(len(test)):
			      samp = test[i]
			      new_samp = []
			      neg_count = 0
			      pos_count = 0
			      for term in samp:
				  if term in neg_pivot_term:
				      new_samp.append(1)
				      '''Detect if there is a negation word before the current word, if there is, reverse its sentiment label'''
				      term_pre1 = ''
				      term_pre2 = ''
				      term_pre3 = ''
				      #term_aft1= ''
				      #term_aft2 = ''
				      if term_idx ==1:
					  term_pre1 = samp[term_idx-1]
				      elif term_idx == 2:
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      elif term_idx >= 3:
					  term_pre3 = samp[term_idx-3]
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      if term_pre1 in adverse_list_idx or term_pre2 in adverse_list_idx or term_pre3 in adverse_list_idx:
					  new_test_label[i][1] = 2
					  pos_count += 1
				      else:
					  new_test_label[i][0] = 2
				          neg_count += 1		
				  elif term in pos_pivot_term:
				      new_samp.append(1)
				      '''Detect if there is a negation word before the current word, if there is, reverse its sentiment label'''
				      term_pre1 = ''
				      term_pre2 = ''
				      term_pre3 = ''
				      #term_aft1= ''
				      #term_aft2 = ''
				      if term_idx ==1:
					  term_pre1 = samp[term_idx-1]
				      elif term_idx == 2:
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      elif term_idx >= 3:
					  term_pre3 = samp[term_idx-3]
					  term_pre2 = samp[term_idx-2]
					  term_pre1 = samp[term_idx-1]
				      if term_pre1 in adverse_list_idx or term_pre2 in adverse_list_idx or term_pre3 in adverse_list_idx:
					  new_test_label[i][0] = 2
					  neg_count += 1
				      else:
					  new_test_label[i][1] = 2
				          pos_count += 1
				  else:
				      new_samp.append(term)
			      if new_test_label[i][0] !=1 or new_test_label[i][1] != 1:
				  test_pivotin_index.append(i)
			      new_test.append(new_samp)
			  new_test = np.array(new_test, dtype=np.int32)
			  new_test_label = np.array(new_test_label, dtype=np.int32)  
			  
			  #The number of test samples, which have pivot words
			  print(len(test_pivotin_index))
			  #The number of development samples, which have pivot words
			  print(len(dev_pivotin_index))
			  #The number of training samples, which do not have pivot words
			  print(len(train_pivotnotin_index))
			
			  # Load word2vec
			  w2v = load_bin_vec(w2v_path, word_to_idx)
			  V = len(word_to_idx) + 1
			  print 'Vocab size:', V
			
			  # Not all words in word_to_idx are in w2v.
			  # Word embeddings initialized to random Unif(-0.25, 0.25)
			  np.random.seed(1515)
			  embed = np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0])))
			  embed[0] = 0
			  for word, vec in w2v.items():
			    embed[word_to_idx[word] - 1] = vec
			
			  # Shuffle train
			  print 'train size:', train.shape
			  N = train.shape[0]
			  perm = np.random.permutation(N)
			  train = train[perm]
			  train_label = train_label[perm]
			  new_train = new_train[perm]
			  new_train_label = new_train_label[perm]
			  
			  output_dir = 'hdf5file'
			  if not os.path.exists(output_dir):
			      os.makedirs(output_dir)			      
			
			  filename = output_dir + os.sep + dataset + '.hdf5'
			  with h5py.File(filename, "w") as f:
			    f["w2v"] = np.array(embed)
			    f['train'] = train
			    f['train_label'] = train_label
			    f['new_train'] = new_train
			    f['new_train_label'] = new_train_label    
			    f['test'] = test
			    f['test_label'] = test_label
			    f['new_test'] = new_test
			    f['new_test_label'] = new_test_label    
			    f['dev'] = dev
			    f['dev_label'] = dev_label
			    f['new_dev'] = new_dev
			    f['new_dev_label'] = new_dev_label			  


if __name__ == '__main__':
  w2v_path = '/home/jfyu/torch/1.bin'
  main(w2v_path)
