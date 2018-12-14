# coding: utf-8
"""Converts symbol csv to indexed one """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys, collections
import tarfile
import os.path
from os import listdir
from os.path import isfile, join
import json, pickle
import re, bz2, random, csv
import nltk.data, csv
from tqdm import tqdm
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.corpus import wordnet
import numpy as np
from itertools import chain
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def make_synonyms(word):
    synonyms = wordnet.synsets(word)
    seq = chain.from_iterable([word.lemma_names() for word in synonyms])
    seen = set()
    seen_add = seen.add
    lemmas = [x for x in seq if not (x in seen or seen_add(x))]
    return lemmas

def remove_punct(sent):
    lsent = tokenizer.tokenize(sent)
    return " ".join(lsent)

def load_vocab_char():
    #mean padding, BOS, EOS, and OOV and <num>
    vocab = u'''␀␂␃⁇N abcdefghijklmnopqrstuvwxyz'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_vocab_word(word_tags):
    #mean padding, BOS, EOS, and OOV
    with open("vocabulary0703.pickle", "rb") as f:
        vocabulary = pickle.load(f)
    # pad BOS EOS OOV
    word_vocab = [] #['\xe2\x90\x80', '\xe2\x90\x82', '\xe2\x90\x83', '\xe2\x81\x87', '<num>']#[u'␀',u'␂',u'␃', u'⁇']
    # threshold = 80 vocab size = 29861
    threshold = 390 # vocab size = 12987
    for word in vocabulary.keys():
        if vocabulary[word] > threshold:
            if word.isdigit(): continue
            if word in word_tags and word_tags[word]=='CD': continue
            if word:
                word_vocab.append(word.lower())
    word_vocab = [u'␀',u'␂',u'␃', u'⁇', '<num>'] + list(set(word_vocab))
    print(len(word_vocab))
    word2idx = {word: idx for idx, word in enumerate(word_vocab)}
    idx2word = {idx: word for idx, word in enumerate(word_vocab)}
    with open("word_vocab0703.pickle",'wb') as f:
        pickle.dump(word_vocab, f)
    return word2idx, idx2word

def convert_sents_toind_char(sents,maxlen, char2idx, word2idx, word_tags): 
    par, Sources = [], []
    numeric = []
    digits = set('0123456789')
    unic  = {'\xe2\x90\x80', '\xe2\x90\x82', '\xe2\x90\x83', '\xe2\x81\x87'}
    unic_dict = {'\xe2\x90\x80':u'␀', '\xe2\x90\x82':u'␂', '\xe2\x90\x83':u'␃', '\xe2\x81\x87':u'⁇'}
    for i, source_sent in enumerate(sents):
        source_sent = remove_punct(source_sent)
        init_sent = source_sent.split()
        sent_list = source_sent.split()
        for ind in range(len(sent_list)):
            if sent_list[ind].isdigit() or (sent_list[ind] in word_tags and word_tags[sent_list[ind]]=='CD'): 
                sent_list[ind] = "N" #unic_dict[sent_list[ind]]
                continue

            if set(sent_list[ind]) & digits:
                lemmas = make_synonyms(sent_list[ind])
                for wrd in lemmas:
                    if wrd in word_tags and word_tags[wrd] in {'OD', 'CD', 'NN'} :
                        sent_list[ind] = wrd
                        break
                if not set(sent_list[ind]) & digits: continue 

            if sent_list[ind] in word2idx and set(sent_list[ind]) & digits:
                word = re.sub(r"[^a-z]", "", sent_list[ind])
                if word in word2idx and word not in {'k', 'x', 'b'}:
                    sent_list[ind] = word
                    continue
                elif word == 'ww':
                    sent_list[ind] = 'war'
                    continue

                sent_list[ind] = "N"
                continue

            if sent_list[ind] not in word2idx:
                sent_list[ind] = u'⁇'


        source_sent = " ".join(sent_list)
        source_sent=re.sub(r"[^a-z]", " ", source_sent)
        x = [char2idx[char] for char in source_sent] # 3: OOV
        if len(x) > maxlen-2: 
            for ind in range(len(init_sent)):
                if set(init_sent[ind]) & digits:
                    if len(init_sent[ind]) < len(sent_list[ind]):
                        sent_list[ind] = "N"

            source_sent = " ".join(sent_list)
            x = [char2idx[char] for char in source_sent] # 3: OOV
            if len(x) > maxlen-2: 
                print(len(x))
                x = x[:maxlen-2]

        x = [1]+x+[2]
        if len(x) < maxlen:
            x += [0] * (maxlen - len(x)) # zero postpadding
        
        Sources.append(source_sent)        
        par.append(x)
        numeric.append(" ".join(map(str,x)))

    return par, Sources, numeric

def convert_sents_toind_word(sents,maxlen, word2idx, word_tags): 
    par, Sources = [], []
    numeric = []
    for i, source_sent in enumerate(sents):
        source_sent = remove_punct(source_sent)
        sent_list = source_sent.split()
        for ind in range(len(sent_list)):
            if sent_list[ind].isdigit() or (sent_list[ind] in word_tags and word_tags[sent_list[ind]]=='CD'): 
                sent_list[ind] = '<num>' #unic_dict[sent_list[ind]]
                continue
            if sent_list[ind] not in word2idx:
                lemmas = make_synonyms(sent_list[ind])
                for wrd in lemmas:
                    if wrd in word2idx:
                        sent_list[ind] = wrd.lower()
                        break
                if sent_list[ind] not in word2idx:
                    sent_list[ind] = u'⁇'

        x = [1]+[word2idx[word] for word in sent_list]+[2] 
        if len(x) < maxlen:
            x += [0] * (maxlen - len(x)) # zero postpadding
        elif len(x) > maxlen:
            x = x[:maxlen-1] + [2]
        Sources.append(" ".join(sent_list))        
        par.append(x)
        numeric.append(" ".join(map(str,x)))
    return par, Sources, numeric

def convert_topairs():
    word_tags = dict()
    for word,pos in brown.tagged_words():
        word_tags[word]=pos

    char2idx, idx2char = load_vocab_char()
    word2idx, idx2word = load_vocab_word(word_tags)

    mydir = 'datasets/changeMV/csv148_pairs0703' # csv147_pairs
    maxlen = 150
    wmaxlen = 25
    subpars_lens = {3:0, 6:0, 10:0, 15:0, 20:0, 30:0, 40:0}
    pars_count = collections.OrderedDict(sorted(subpars_lens.items(), key=lambda t: t[0]))
    max_lenfile = 25000
    ffiles = [f for f in listdir(mydir) if isfile(join(mydir, f))]
    files = []
    for el in ffiles:
        #print(el)
        if el.split('.')[0].isdigit():
            files.append(el)
    onlyfiles = sorted(files, key=lambda x: int(x.split('.')[0]))
    symb_file = 'datasets/changeMV/csveq_0703/s148_0703'
    num_file = 'datasets/changeMV/csveq_0703/numb148_0703'
    if not os.path.exists(symb_file):
        os.makedirs(symb_file)
    if not os.path.exists(num_file):
        os.makedirs(num_file)


    for i,el in tqdm(enumerate(onlyfiles)):
        fl = os.path.join(mydir,el)
        print(fl)
        with open(fl, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for indp, paragraphs in tqdm(enumerate(reader)):
                #print(paragraphs)
                ls = [paragraphs[0].lower().split('&'), paragraphs[1].lower().split('&')]
                if len(ls[0])==0 or len(ls[1])==0 or ls[0]=='removed' or ls[1] == 'removed':
                    continue
                mx = max(len(ls[0]), len(ls[1]))
                ids1, src1, numeric1 = convert_sents_toind_char(ls[0],maxlen,char2idx, word2idx,word_tags)
                ids2, src2, numeric2 = convert_sents_toind_char(ls[1],maxlen,char2idx, word2idx,word_tags)
                if len(ls[1]) == mx:
                    for lj in range(mx - len(ls[0])):
                        numeric1.append(" ".join(map(str,[0]*(maxlen))))
                else:
                    for lj in range(mx - len(ls[1])):
                        numeric2.append(" ".join(map(str,[0]*(maxlen))))

                _, _, wnumeric1 = convert_sents_toind_word(ls[0],wmaxlen,word2idx,word_tags)
                _, _, wnumeric2 = convert_sents_toind_word(ls[1],wmaxlen,word2idx,word_tags)
                if len(ls[1]) == mx:
                    for lj in range(mx - len(ls[0])):
                        wnumeric1.append(" ".join(map(str,[0]*(wmaxlen))))
                else:
                    for lj in range(mx - len(ls[1])):
                        wnumeric2.append(" ".join(map(str,[0]*(wmaxlen))))
                
                for length in pars_count.items():
                    if mx<=length[0]:
                        pars_count[length[0]]+=1
                        fileid = pars_count[length[0]]//max_lenfile                                                       
                        fl1 = os.path.join(symb_file,'%d_%d.csv' % (fileid,length[0]))
                        fd1 = open(fl1,'a')
                        writer1 = csv.writer(fd1)
                        fl2 = os.path.join(num_file,'%d_%d.csv' % (fileid,length[0]))
                        fd2 = open(fl2,'a')
                        writer2 = csv.writer(fd2)
                        #store symbolic paragraph
                        writer1.writerow([paragraphs[0],paragraphs[1]])                        
                        #store numeric paragraph
                        writer2.writerow(["|".join(numeric1), "|".join(numeric2),"|".join(wnumeric1), "|".join(wnumeric2)])
                        fd1.close() 
                        fd2.close() 
                        break                    

            if i%30==0:
                print(pars_count)
                print (ls,"|".join(numeric1))
                print (ls,"|".join(numeric2))
                print (ls,"|".join(wnumeric1))
                print (ls,"|".join(wnumeric2))
    
    fd1.close()
    fd2.close()
    print(pars_count)


def main():
    # Get the data.
    convert_topairs()

main()



