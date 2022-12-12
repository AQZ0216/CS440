# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.025, pos_prior=0.8,silently=False):
    print_paramter_vals(laplace,pos_prior)

    cnt_pos = Counter()
    cnt_neg = Counter()

    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            if train_labels[i] == 1:
                cnt_pos[train_set[i][j]] += 1
            else:
                cnt_neg[train_set[i][j]] += 1

    n_pos = sum(cnt_pos.values())
    n_neg = sum(cnt_neg.values())
    V_pos = len(list(cnt_pos))
    V_neg = len(list(cnt_neg))
    
    pos_dict = {}
    neg_dict = {}

    for key, value in cnt_pos.items():
        pos_dict[key] = math.log(value+laplace) - math.log(n_pos+laplace*(V_pos+1))

    for key, value in cnt_neg.items():
        neg_dict[key] = math.log(value+laplace) - math.log(n_neg+laplace*(V_neg+1))

    UNK_pos = math.log(laplace) - math.log(n_pos+laplace*(V_pos+1))
    UNK_neg = math.log(laplace) - math.log(n_neg+laplace*(V_neg+1))

    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        pos = math.log(pos_prior)
        neg = math.log(1-pos_prior)
        for i in range(len(doc)):
            pos += pos_dict.get(doc[i], UNK_pos)
            neg += neg_dict.get(doc[i], UNK_neg)

        if pos > neg:
            yhats.append(1)
        else :
            yhats.append(0)
        
    return yhats





def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.025, bigram_laplace=0.005, bigram_lambda=0.4,pos_prior=0.8, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    cnt_pos = Counter()
    cnt_neg = Counter()
    cnt_pos_bigram = Counter()
    cnt_neg_bigram = Counter()

    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            if train_labels[i] == 1:
                cnt_pos[train_set[i][j]] += 1

                if j+2 <= len(train_set[i]):
                    cnt_pos_bigram[tuple(train_set[i][j:j+2])] += 1
            else:
                cnt_neg[train_set[i][j]] += 1

                if j+2 <= len(train_set[i]):
                    cnt_neg_bigram[tuple(train_set[i][j:j+2])] += 1

    n_pos = sum(cnt_pos.values())
    n_neg = sum(cnt_neg.values())
    V_pos = len(list(cnt_pos))
    V_neg = len(list(cnt_neg))
    n_pos_bigram = sum(cnt_pos_bigram.values())
    n_neg_bigram = sum(cnt_neg_bigram.values())
    V_pos_bigram = len(list(cnt_pos_bigram))
    V_neg_bigram = len(list(cnt_neg_bigram))   
    
    pos_dict = {}
    neg_dict = {}
    pos_dict_bigram = {}
    neg_dict_bigram = {}

    for key, value in cnt_pos.items():
        pos_dict[key] = math.log(value+unigram_laplace) - math.log(n_pos+unigram_laplace*(V_pos+1))

    for key, value in cnt_neg.items():
        neg_dict[key] = math.log(value+unigram_laplace) - math.log(n_neg+unigram_laplace*(V_neg+1))

    for key, value in cnt_pos_bigram.items():
        pos_dict_bigram[key] = math.log(value+bigram_laplace) - math.log(n_pos_bigram+bigram_laplace*(V_pos_bigram+1))

    for key, value in cnt_neg_bigram.items():
        neg_dict_bigram[key] = math.log(value+bigram_laplace) - math.log(n_neg_bigram+bigram_laplace*(V_neg_bigram+1))

    if unigram_laplace == 0:
        UNK_pos = 0
        UNK_neg = 0
    else:
        UNK_pos = math.log(unigram_laplace) - math.log(n_pos+unigram_laplace*(V_pos+1))
        UNK_neg = math.log(unigram_laplace) - math.log(n_neg+unigram_laplace*(V_neg+1))
    UNK_pos_bigram = math.log(bigram_laplace) - math.log(n_pos_bigram+bigram_laplace*(V_pos_bigram+1))
    UNK_neg_bigram = math.log(bigram_laplace) - math.log(n_neg_bigram+bigram_laplace*(V_neg_bigram+1))

    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        pos = math.log(pos_prior)
        neg = math.log(1-pos_prior)
        pos_bigram = math.log(pos_prior)
        neg_bigram = math.log(1-pos_prior)
        for i in range(len(doc)):
            pos += pos_dict.get(doc[i], UNK_pos)
            neg += neg_dict.get(doc[i], UNK_neg)

            if i+2 <= len(doc):
                pos_bigram += pos_dict_bigram.get(tuple(doc[i:i+2]), UNK_pos_bigram)
                neg_bigram += neg_dict_bigram.get(tuple(doc[i:i+2]), UNK_neg_bigram)

        if pos*(1-bigram_lambda)+pos_bigram*bigram_lambda > neg*(1-bigram_lambda)+neg_bigram*bigram_lambda:
            yhats.append(1)
        else :
            yhats.append(0)
    return yhats

