import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import random
from pathlib import Path
from time import time
from datetime import datetime
from pprint import pprint
from warnings import warn
from datetime import datetime
import itertools
from collections import Counter, OrderedDict
from bidi import algorithm as bidi
from hebrew_stopwords import hebrew_stopwords


EXTRA_STOPWORDS = (
    'לך',
)


WORDS_SEPS = ' | - |–|\t|\n\r|\n'
CHARS_TO_FILTER = '\.|,|"|\(|\)|;|:|\?|\t'


def ordered_counter(tokens):
	return OrderedDict(sorted(Counter([tok for grp in tokens for tok in grp]).items(),
                              key=lambda kv: kv[1], reverse=1))

def tokenize(texts, seps=WORDS_SEPS, chars_to_filter=CHARS_TO_FILTER,
             stopwords=hebrew_stopwords, filter_fun=None, apply_fun=None):
    
    words = [re.split(seps,txt) for txt in texts]
    words = [re.sub(chars_to_filter, '', w.strip())
             for text_words in words for w in text_words]
    words = [w for w in words if w and w not in stopwords]
    
    if filter_fun:
        words = [w for w in words if filter_fun(w)]
    if apply_fun:
        words = [apply_fun(w) for w in words]
        
    return words


class Parser:
    def __init__(self, df, meta=None,
                 words_seps=WORDS_SEPS, chars_to_filter=CHARS_TO_FILTER,
                 filter_fun=lambda s: len(s)>1 or '0'<=s<='9', apply_fun=None,
                 stopwords=hebrew_stopwords+EXTRA_STOPWORDS,
                 max_voc_size=1000, min_voc_occurences=6,
                 laplace_smooth=0,
                 do_init=True):
        # data
        self.meta = meta
        self.df = df
        self.df.body.fillna('', inplace=True)
        self.df.header.fillna('', inplace=True)
        # tokenization conf
        self.words_seps = words_seps
        self.chars_to_filter = chars_to_filter
        self.filter_fun = filter_fun
        self.apply_fun = apply_fun # TODO use this for lemmatization
        self.stopwords = stopwords
        self.tokens = None
        # vocabulary
        self.max_voc_size = max_voc_size
        self.min_voc_occurences = min_voc_occurences
        self.vocabulary = None
        # one hot encoding
        self.laplace_smooth = laplace_smooth
        self.one_hot = None
        
        if do_init:
            self.tokenize()
            self.set_vocabulary()
            self.set_one_hot_embedding()
    
    # Parsing
    def tokenize(self, by='ID'):
        self.tokens = self.df.groupby(by).apply(lambda d: tokenize(d.body.values, self.words_seps, self.chars_to_filter,
                                                                   self.stopwords, self.filter_fun, self.apply_fun))
    
    def set_vocabulary(self):
        all_tokens = [tok for group in self.tokens for tok in group]
        counter = OrderedDict(sorted(Counter(all_tokens).items(), key=lambda kv: kv[1], reverse=1))
        voc = list(counter.keys())
        if self.max_voc_size:
            voc = voc[:self.max_voc_size]
        if self.min_voc_occurences:
            i = int(np.sum(np.array(list(counter.values()))>=self.min_voc_occurences))
            voc = voc[:i]
        self.vocabulary = voc
            
    def set_one_hot_embedding(self):
        counters = {w: len(self.tokens)*[self.laplace_smooth] for w in self.vocabulary}
        for i,group in enumerate(self.tokens):
            for w in group:
                if w in counters:
                    counters[w][i] += 1
        self.one_hot = pd.DataFrame({**counters})
    
    # Analysis
    def full_protocol(self, ID=None):
        if ID is None:
            ID = random.choice(np.unique(self.df.ID))
        return '\n\n'.join(self.df[self.df.ID==ID].body.values)
    
    def show_tokens(self, **kwargs):
        show_freqs([tok for group in self.tokens for tok in group], **kwargs)
        
    def one_hot_pca(self):
        # TODO plot PCA of one_hot with 2 components
        warn('PCA display of protocols is not implemented :(')
		

def show_freqs(tokens, vocabulary=None, n=15, logscale=True, token_lab='Token', compact=False):
    if compact:
        _, axs = plt.subplots(1,3, figsize=(18,5))
    else:
        _, axs = plt.subplots(3,1, figsize=(12,12))
    
    if vocabulary:
        vocabulary = set(vocabulary)
        tokens = [token for token in  tokens if token in vocabulary]
    count = Counter(tokens)
    count = OrderedDict(sorted(count.items(), key=lambda kv: kv[1]))
    vals = list(count.values())
    keys = list(count.keys())
    bidi_keys = [bidi.get_display(token) for token in keys]
    
    # quantile plot
    ax = axs[0]
    ax.axhline(np.sum(vals)/len(vals), linestyle=':', color='blue', label='Average')
    ax.plot(list(range(101)), [vals[int(q/100*(len(vals)-1))] for q in range(101)], 'k.-')
    ax.set_xlabel('Quantile [%]')
    ax.set_ylabel('Number of occurences')
    ax.set_xlim((0,100))
    ax.set_title(f'Total: {len(tokens):d}, Unique: {len(keys):d}')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()
    
    # top tokens
    ax = axs[1]
    ax.bar(bidi_keys[-1:-n-1:-1], vals[-1:-n-1:-1])
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel(token_lab)
    ax.set_ylabel('Occurences')
    ax.set_title('Top')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    for label in ax.xaxis.get_ticklabels():
        label.set_ha('right')
    
    # tail tokens
    ax = axs[2]
    ax.bar(list(reversed(bidi_keys[:n])), list(reversed(vals[:n])))
    ax.tick_params(axis='x', rotation=45)
    for label in ax.xaxis.get_ticklabels():
        label.set_ha('right')
    ax.set_xlabel(token_lab)
    ax.set_ylabel('Occurences')
    ax.set_title('Tail')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    
    plt.tight_layout()
    
    return (list(reversed(keys)), list(reversed(vals)))
