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
	lemmas = [x.lower() for x in seq if not (x in seen or seen_add(x))]
	return lemmas

emb_dim = 300
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)


def load_word_vocab():
    with open("word_vocab0703.pickle", "rb") as f:
        word_vocab = pickle.load(f)
            # pad BOS EOS OOV Numbers
    word_vocab =[u'␀',u'␂',u'␃', u'⁇', 'N'] + word_vocab[5:]
    word2idx = {char: idx for idx, char in enumerate(word_vocab)}
    idx2word = {idx: char for idx, char in enumerate(word_vocab)}
    return word2idx, idx2word


word2ind, ind2word = load_word_vocab()
print("vocab size = %d" %len(word2ind))

word_vocab = ind2word.values()
s = ""
for word in word_vocab[5:]:
	s += word + '\n'
s = s.strip()
with open("vocab_ascii.txt", 'w') as txtfile:
	txtfile.write(s)

word_emb_glove = np.loadtxt("glove300d_0722.txt")

embeddings = np.random.rand(len(word_vocab), emb_dim)
embeddings[:5] = word_emb_glove[:5]
count=0
for i in tqdm(range(5, len(word_vocab))):
	if word_vocab[i] in model.wv:
		embeddings[i] = model.wv[word_vocab[i]]
	else:
		word = word_vocab[i].lower()
		lemmas = make_synonyms(word)
		if lemmas != []:
			for wrd in lemmas:
				if wrd in model.wv:
					word = wrd.lower()
					break
			#print(lemmas)
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
		
		if word in model.wv:
			embeddings[i] = model.wv[word]
		else:
			print(word_vocab[i], lemmas)
			embeddings[i] = word_emb_glove[i]
			count +=1
print(count)

np.savetxt("gnews300d_0722.txt", embeddings)