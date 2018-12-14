# coding: utf-8
import gensim, pickle, re, enchant
from nltk.corpus import wordnet
import numpy as np
from itertools import chain
from tqdm import tqdm

from nltk.metrics import edit_distance

class SpellingReplacer(object):
    def __init__(self, dict_name = 'en_GB', max_dist = 2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = 2

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

def make_synonyms(word):
	synonyms = wordnet.synsets(word)
	seq = chain.from_iterable([word.lemma_names() for word in synonyms])
	seen = set()
	seen_add = seen.add
	lemmas = [x for x in seq if not (x in seen or seen_add(x))]
	return lemmas

emb_dim = 300
model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.300d.txt',binary=False)

with open("word_vocab0703.pickle", "rb") as f:
        word_vocab = pickle.load(f)
    # pad BOS EOS OOV
print("vocab size = %d" %len(word_vocab))

embeddings = np.random.rand(len(word_vocab), emb_dim)
count=0
for i in tqdm(range(5, len(word_vocab))):
	if word_vocab[i] in model.wv:
		embeddings[i] = model.wv[word_vocab[i]]
	else:
		print(word_vocab[i])
		word = word_vocab[i].lower()
		lemmas = make_synonyms(word)
		if lemmas != []:
			for wrd in lemmas:
				if wrd in model.wv:
					word = wrd.lower()
					break
			print(lemmas)
			if word in model.wv:
				embeddings[i] = model.wv[word]

		if not word_vocab[i].isalpha():
			word = re.sub(r"[^a-z]", "", word)
		if word == "": 
			count += 1
			continue
		elif word in model.wv:
			embeddings[i] = model.wv[word]
			continue
		replacer = SpellingReplacer()
		word = replacer.replace(word)
		lemmas = make_synonyms(word)
		for wrd in lemmas:
			if wrd in model.wv:
				word = wrd.lower()
				break
		print(lemmas)
		if word in model.wv:
			embeddings[i] = model.wv[word]
		else:
			count +=1
print(count)

np.savetxt("glove300d_0704.txt", embeddings)