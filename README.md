Author

Jianfei YU
jfyu.2014@phdis.smu.edu.sg

Feb 22, 2017

# Data and Code for:

Learning Sentence Embeddings with Auxiliary Tasks for Cross-Domain Sentiment Classification
EMNLP 2016
https://aclweb.org/anthology/D/D16/D16-1023.pdf

## Requirements

1. Python 2.7
2. hdf5 (h5py)
3. Torch7

## I. Data

1. The data consists of five domains: the first two are movie domains, the latter three are respectively digital device, laptop and restaurant.

2. The following is the summarization of all the data

| Datasets        | Sentences | Words |
| -------------   |:---------:| -----:|
| rt-polarity(MV1)|10662      |  18765|
| stsa.binary(MV2)| 9613      |  16186|  
| custrev(CR)     | 3770      |   5340|
| laptop(LT)      | 1907      |   2837|   
| restaurant(RT)  | 1572      |   2930|


       
3. Each domain's data is under the folder "preprocess codes//data", and the data with Part-of-speech taggings are under the folder "preprocess codes//tag_data".



## II. Evaluation 

   For evaluation, we treat every source-target domain pair as a task, and hence have 18 tasks (The first two are not considered since they are both from movie domains).
   
   In our experiments, we use a Transductive Learning setting.
   
   That is, we use the unlabeled data from both the development set and the test set in the target domain to train our model, and we only tune parameters on the development set in the target domain.
   
   For example, assume that the task is MV2->LT,
   
   During training stage, we will use labeled data from MV2 (i.e., stsa.binary.txt), and unlabeled data from LT (i.e., laptop.txtdev and laptop.txttest). Note that we cannot make use of labels from LT during training stage
   
   During test stage, we will use the trained model to tune on the development set (i.e., laptop.txtdev), and apply the best tuned model to test on the test set (i.e., laptop.txttest) 
 


 
## III. Code

### Part1: Pre-process code (under the folder "preprocess codes"): 

Run on Python 2.7, and the pipeline pre-process code requires Python package hdf5 (the h5py module)

##### Step 1. Obtain Pivot Word List

```
python IG2.py
```

##### Step 2. The pivot word lists of 18 source-target pairs will be in the folder "pivotlist", and then you can run the following codes:

```
python preprocess_dasa_mt_2_2.py
```

Note that before you run, you need to download word2vec vectors from here: https://code.google.com/archive/p/word2vec/  , and then set w2v_path in line 466.

### Part2: Model Code:

Run on Torch7, and the training pipeline requires the lua package: hdf5

To run the baseline (NaiveNN in our paper), you can just run:

```
sh run1.sh
```

To run our Joint model (Joint in our paper), you can just run:

```
sh run2.sh
```

----------------------------------------training details----------------------------------------------------------

Note that here we employ an alternating optimization approach for training our Joint model.

In each epoch, we first use labeled source domain data to optimize all the model parameters (one sentence, one true label, two auxiliary labels), and then switch to using unlabeled target domain data to optimize the parameters corresponding to the auxiliary task (one sentence, two auxiliary labels).


## Acknowledgements

Most of the code are based on the code by Harvard NLP group: https://github.com/harvardnlp/sent-conv-torch.

Using this code means you have read and accepted the copyrights set by the dataset providers.

# Licence:

Singapore Management University
