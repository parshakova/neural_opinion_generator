# coding: utf-8
# 06.20
# save from json and count number of oversized sentences
# len=150
# replace unfrequent words by <unk>

import tarfile
import os.path, pickle
import json
from tqdm import tqdm
import re, bz2, random, csv
import nltk.data, csv, nltk
from collections import Counter
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


def clean_str_char(string):
    # vocab = u'''␀␂␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789().?!,:'-`;'''
    string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', string)
    string = re.sub(r"\"", "\'", string)
    string = re.sub(r"[^a-z0-9(),;:\-!.?\']", " ", string)     
    string = re.sub(r'[;:\?\.\!\s]+(?=[\?\.\!])', '', string)
    string = re.sub(r'[;:\-\s]+(?=[;:\-])', '', string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"\.", ". ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r",{2,}", ",", string)
    string = re.sub(r"\n", "", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()

def prepro(s):
    s = s.lower()
    i0 = re.search(r"EDIT: ", s)
    if i0 != None:
        s = s[:i0.start()]
    i1 = re.search(r"_____", s)
    if i1 != None:
        s1 = clean_str_char(s[:i1.start()])
    else: 
        s1 = clean_str_char(s)
    return s1

def remove_punct(sent):
    lsent = tokenizer.tokenize(sent)
    return " ".join(lsent)


def remove_brack(el):
    prev = [0,0,0]
    inds = [-1,len(el)-1, -1, len(el)-1]
    if re.search('\(',el) == None:
        return False
    for ind in range(len(el)):
        if el[ind] == '(':
            if prev[0] == 0:
                inds[0] = ind
            prev[0] += 1
        elif el[ind] == ')':
            prev[0] -= 1
            if prev[0] == 0:
                inds[1] = ind
                if prev[1] > prev[2]:
                    prev[2] = prev[1]
                    inds[2] = inds[0]
                    inds[3] = inds[1]
                prev[1] = 0
        else:
            if prev[0] != 0:
                prev[1] += 1  
    if prev[1] != 0 or prev[2] != 0:     
        if inds[2] == -1 or prev[1] > prev[2]:
            s = el[:inds[0]]   
            if inds[1] != len(el)-1 and inds[0]<inds[1]:
                s += el[inds[1]+1:]
        else:
            s = el[:inds[2]]   
            if inds[3] != len(el)-1:
                s += el[inds[3]+1:]  
        s1 = s.strip()
        s1 = prepro(s1)
        if s1 == "":
            return ("","")
        if s1[-1].isalnum() or s1[-1] == ')':
            s1 += '.'
        else:
            i = len(s1)-1
            while i >= 1 and not s1[i].isalnum():
                i -= 1
            s1 = s1[:i+1] + '.'
        return (s1,"")
    else:       
        return False
    
def wrap_endings(el,ind):
    #print("ind   %d  el   ~~~~~  %s      "%(ind,el))
    s1 = el[:ind].strip()
    if ind >0 and s1[-1].isalnum() or s1[-1] == ')':
        s1 += '.'
    elif ind >0:
        i = len(s1)-1
        while i >= 1 and not s1[i].isalnum():
            i -= 1
        s1 = s1[:i+1] + '.' 
    if el[ind].isalnum():
        s2 = el[ind].upper() + el[ind+1:]
    else:
        i = ind+1
        while i < len(el)-1 and not (el[i].isalnum() or el[i]=='('):
            i += 1
        s2 = el[i].upper()+el[i+1:]
    return (s1, s2)

def wrap_beginning(s2):
    s2 = s2.strip()
    if len(s2) == 1 or len(s2) == 0:
        return ""
    if s2[0].isalnum():
        s2 = s2[0].upper() + s2[1:]
    else:
        i = 1
        while i < len(s2)-1 and not (s2[i].isalnum() or s2[i]=='('):
            i += 1
        s2 = s2[i].upper()+s2[i+1:]
    s2 = re.sub(r'(?<=[\?\.\!])[\s\)]+', '', s2)
    return s2
    
        
def check_for_separ(separ, el, prec=None):
    prev = (-1,-1)
    el = el.strip()
    for i,sep in enumerate(separ):
        sub_el = el
        ind = re.search(sep, sub_el)
        s_cum = 0
        while ind != None:
            e = ind.end()-1
            s = ind.start()
            if prec is not None:
                pred_pos = re.search(prec[i],sub_el[:e])
                if pred_pos != None and pred_pos.end() == s -1:
                    sub_el = sub_el[e+1:]
                    s_cum += e+1
                    ind = re.search(sep, sub_el)
                    continue                
            if  s+s_cum > 0 and abs(s+s_cum-len(el)/2)<abs(prev[1]-len(el)/2) and s+s_cum < len(el) - len(sep):
                prev = (i, s+s_cum)
                
            sub_el = sub_el[e+1:]
            s_cum += e+1
            ind = re.search(sep, sub_el)
    if prev[1] != -1:                    
        return wrap_endings(el,prev[1])
    else:        
        return False
    
def make_shorter(el):
    # need to split a sentence
    out = remove_brack(el)
    if out == False:
        separ = ['nevertheles', 'nonetheless','however', ' but ', ' still ', 'yet ', ' though ', 'although',\
                 'even so', 'for all that', 'despite that', 'in spite of that', 'anyway', 'anyhow',\
                 'as long as', ' and that', ' and thus','because','since', ',but ', ',and that']
        out = check_for_separ(separ, el)
        if out == False:
            out = check_for_separ([' so '], el,"that")
            if out == False:
                out = check_for_separ([':',';','- '],el)
                if out == False:
                    out = check_for_separ([','],el,None)
                    if out == False:
                        out = check_for_separ([' and '],el,None)
                        if out == False:
                            out = check_for_separ(['at which'],el)
                            if not out:
                                out = check_for_separ(['while', 'which', 'I mean'],el,None)
                                if not out:
                                    return False                      
    return out

def pos_tag_nltk(el):
    vocb = {'NN','NNS','NNP','NNPS','RBS','PRP'}
    l2 = el.split()
    tags = nltk.pos_tag(l2);
    # select all items within the range from vocab
    ars = []
    for i, it in enumerate(tags):
        if it[1] == 'DT': ars.append(i)
    if ars == [] or len(ars)<=3:
        for i, it in enumerate(tags):
            if it[1] in vocb: ars.append(i)
    if ars == [] or len(ars)<=3:
        ind = min(max(2,random.randint(len(l2)/3,2*len(l2)/3)), len(l2)-3)
        (s1,s2) = wrap_endings(el, len(" ".join(l2[:ind]))+1)
    else:
        ars.sort()
        ind = min(max(2,random.randint(len(ars)//3,2*len(ars)//3)), len(ars)-3)
        #print("arsind %d ind %d"%(ars[ind],ind), list(map(lambda i: l2[i],ars)))
        (s1,s2) = wrap_endings(el, len(" ".join(l2[:ars[ind]])))
    return (s1, s2)



def split_on_sentences(row, mark=False):
        l = sent_detector.tokenize(row.strip())
        if mark:
            print('\n\n')
            print l
            print('\n')
        bad, all_s = 0, 0
        paragraph = []
        max_length = 148 # total with <BOS> and <EOS> length = 150
        for el in l:
            el = re.sub(r'(?<=[\?\.\!])[\s\(]+', '', el)            
            if len(remove_punct(el)) <= max_length:
                el =  wrap_beginning(remove_punct(el))
                paragraph.append(remove_punct(el))
            else:
                ret = make_shorter(el)                
                ar = []
                count = 0
                if ret!= False: queue = [ret[0],ret[1]] 
                else:
                    s1, s2 = pos_tag_nltk(el)
                    queue = [s1, s2]
                    bad += 2   

                while queue != []:
                    sent = queue[0]
                    #print(len(sent))
                    count += 1
                    if sent =="":
                        queue.pop(0)
                        continue
                    if len(remove_punct(sent)) <= max_length and sent != "":
                        ar.append(remove_punct(sent))
                        queue.pop(0)
                        #print("shoooooooooorter", queue)
                        continue
                    else:
                        ret = make_shorter(sent)
                        if ret == False: 
                            queue.pop(0)
                            s1, s2 = pos_tag_nltk(sent)
                            queue = [s1, s2] + queue
                            bad += 2
                            continue
                        queue.pop(0)
                        if len(remove_punct(ret[0])) <= max_length:
                            if len(remove_punct(ret[1])) <= max_length:
                                ar += [remove_punct(ret[0]),remove_punct(ret[1])]
                            else:
                                ar += [ret[0]]
                                if ret[1] != "": queue = [ret[1]] + queue
                        else:
                            if ret[1] != "": queue = [ret[1]] + queue
                            if ret[0] != "": queue = [ret[0]] + queue
                        if count > 100:
                            break
                for sent in ar:
                    sent =  wrap_beginning(sent)
                    if len(remove_punct(sent)) > max_length:
                        print("########################",remove_punct(sent))
                        break
                    if sent != "":                        
                        paragraph.append(remove_punct(sent))

        ret = "&".join(paragraph)
        if mark:
            for el in paragraph:
                print el
        return (ret, bad, len(paragraph))
            
from os import listdir
from os.path import isfile, join
mydir = '/home/tanichka/Documents/datasets/changeMV/json_data_250'
files = [f for f in listdir(mydir) if isfile(join(mydir, f))]
onlyfiles = sorted(files, key=lambda x: int(x.split('.')[0]))

fpairs = '/home/tanichka/Documents/datasets/changeMV/csv148_pairs0703'
if not os.path.exists(fpairs):
    os.makedirs(fpairs)

Bad, Total, Pars = 0,0, 0


with open("vocabulary0703.pickle", 'rb') as f:
    vocabulary = pickle.load(f)


sentence_length = Counter()
for i,el in tqdm(enumerate(onlyfiles)):
    path = os.path.join(mydir, el)
    fl = os.path.join('/home/tanichka/Documents/datasets/changeMV/csv148_pairs0703','%d.csv' % (i))
    fl1 = os.path.join(fpairs,'%d.csv' % (i))
    fd1 = open(fl1,'a')
    writer1 = csv.writer(fd1)
    print(path)
    all_bad, total = 0,0
    mark = False
    pars = 0
    #sample = "I get the discomfort, the frustration, but I also understand that to deny that the umbrella of white privilege does not apply to all white people is another injustice to struggling minorities who do work harder than I would have to in order to reach the same level of economic stability or respect."
    #par,bad,alls = split_on_sentences(prepro(sample),True)
    
    with open(path,'r') as f1:
        mdata = [json.loads(line.decode('utf-8')) for line in f1 ]
    for item in tqdm(mdata[0]):
        if item['selftext']:
            par,bad,alls = split_on_sentences(prepro(item['selftext']),mark)
            all_bad += bad
            total += alls
            pars += 1
        comms = [par]
        for coms in item['comments']:
            if 'body' in coms:
                ind = re.search("Confirmed: 1 delta awa",coms['body'])
                if ind == None:
                    par,bad,alls = split_on_sentences(prepro(coms['body']))
                    if par!='Deleted' or par!='Removed':
                        all_bad += bad
                        total += alls
                        pars += 1
                        comms.append(par)
        if len(comms) >= 2:                
            for cind in range(len(comms)-1):
                comm_ar = []
                for ind in [cind, cind+1]:
                    temp = []
                    for sent in comms[ind].split("&"):
                        l = sent.split()
                        res_sent = []
                        for word in l:
                            res_sent.append(word)

                        temp.append(" ".join(res_sent)) 
                        sentence_length[len(temp[-1].split())] += 1
                    comm_ar.append("&".join(temp))
                writer1.writerow([comm_ar[0], comm_ar[1]])

        Bad += all_bad
        Total += total
        Pars += pars
    fd1.close()
print(comm_ar)
print(sentence_length)
print "\n               FINAL             \n"
print("oversized:   %d, total:  %d,  ratio:  %d,  paragraphs:  %d"%(Bad, Total,Total/Bad, Pars))


