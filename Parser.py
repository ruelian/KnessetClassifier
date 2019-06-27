import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle as pkl
import random
from pathlib import Path
from time import time
from datetime import datetime
from pprint import pprint
from warnings import warn
from datetime import datetime
from tqdm import tqdm
import itertools
from collections import Counter, OrderedDict
from bidi import algorithm as bidi
from hebrew_stopwords import hebrew_stopwords

'''
TODO Parallel computing (e.g. for large tokenizations):

from concurrent.futures import ProcessPoolExecutor

    if distributed:
        tpool = ProcessPoolExecutor(max_workers=distributed)
        tmp = tpool.map(fun, input_list)
        x = list(tmp)
    else:
        x = [fun(p) for p in input_list]
        
_______________________

TODO consider holding one_hot encoding as count[doc][word] rather than count[word][doc]
without initializing all words (to avoid allocating the whole sparse matrix docsXwords).
This depends on how the future clustering/classifier will expect to have its input. 
'''

EXTRA_STOPWORDS = (
    'לך',
    'רוצה',
    'צריך',
    'הכנסת',
    'היום',
    'משרד',
    'שיש',
    'שזה',
    'דבר',
    'אומר',
    'החוק',
    'הזאת',
    'לגבי',
    'ואני',
    'לעשות',
    'שאנחנו',
    'האלה',
    'חוק',
    'הוועדה',
    'אפשר',
    'שאני',
    'כמה',
    'עכשיו',
    'חושב',
    'הרבה',
    'אלא',
    'לפי',
    'וגם',
    'סעיף',
    'האם',
    'שהם',
    'נכון',
    'ראש',
    'המדינה',
    'רוצים',
    'הממשלה',
    'יודע',
    'ועדת',
    'הדברים',
    'הנושא',
    'בנושא',
    'היור',
    'חבר',
    'צריכים',
    'שהיא',
    'דברים',
    'אנשים',
    'וזה',
    'אומרת',
    'תודה',
    'קודם',
    'באמת',
    'להגיד',
    'לקבל',
    'שר',
    'משהו',
    'לזה',
    'לומר',
    'אדוני',
    'טוב',
    'לכל',
    'בחוק',
    'שום',
    'מבקש',
    'הדיון',
    'שצריך',
    'מדובר',
    'נושא',
    'ואנחנו',
    'מדברים',
    'חלק',
    'באופן',
    'הצעת',
    'בסדר',
    'חברי',
    'הייתי',
    'חשוב',
    'יודעים',
    'בעד',
    'שאתה',
    'והוא',
    'שאין',
    'עושים',
    'בכלל',
    'בבקשה',
    'הדבר',
    'העניין',
    'בצורה',
    'בזה',
    'בעצם',
    'ויש',
    'ואז',
    'בסעיף',
    'מדבר',
    'במקום',
    'ולכן',
    'צריכה',
    'כרגע',
    'כאלה',
    'בעניין',
    'רבה',
    'חושבת',
    'בדיוק',
    'רואה',
    'יהיו',
    'ידי',
    'עדיין',
    'השר',
    'ברור',
    'שאם',
    'מבין',
    'אפילו',
    'אמר',
    'כלומר',
    'פי',
    'הישיבה',
    'לדבר',
    'שגם',
    'אומרים',
    'להגיע',
    'מישהו',
    'ואם',
    'למשל',
    'עושה',
    'שיהיה',
    'כמובן',
    'שכל',
    'אמרתי',
    'שיהיה',
    'בא',
    'הזו',
    'הייתה'
)


WORDS_SEPS = ' | - |–|\t|\n\r|\n'
CHARS_TO_FILTER = '\.|,|"|\(|\)|;|:|\?|\t' # TODO should "'" be included?


def ordered_counter(tokens):
    return OrderedDict(sorted(Counter([tok for grp in tokens for tok in grp]).items(),
                              key=lambda kv: kv[1], reverse=1))

def tokenize(texts, seps=WORDS_SEPS, chars_to_filter=CHARS_TO_FILTER,
             stopwords=hebrew_stopwords, filter_fun=None, apply_fun=None):
    if type(stopwords) is not set:
        stopwords = set(stopwords)
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
    def __init__(self, df=None, meta=None,
                 words_seps=WORDS_SEPS, chars_to_filter=CHARS_TO_FILTER,
                 filter_fun=lambda s: (len(s)>1 or '0'<=s<='9' or 'A'<=s<='Z'), apply_fun=None,
                 stopwords=hebrew_stopwords+EXTRA_STOPWORDS,
                 max_voc_size=1000, min_voc_occurences=10,
                 laplace_smooth=0,
                 do_init=1):
        # data
        self.meta = meta
        self.df = df
        if self.df:
            self.df.body.fillna('', inplace=True)
            self.df.header.fillna('', inplace=True)
        # tokenization conf
        self.words_seps = words_seps
        self.chars_to_filter = chars_to_filter
        self.filter_fun = filter_fun
        self.apply_fun = apply_fun # TODO use this for lemmatization
        self.stopwords = set(stopwords)
        self.tokens = None
        self.profile = None
        # vocabulary
        self.max_voc_size = max_voc_size
        self.min_voc_occurences = min_voc_occurences
        self.vocabulary = None
        # one hot encoding
        self.laplace_smooth = laplace_smooth
        self.one_hot = None

        if do_init:
            if do_init > 1:
                self.tokenize() # hold list of tokens per text - less efficient
            else:
                self.corpus_profile()
            self.set_vocabulary()
            self.set_one_hot_embedding()

    # Parsing
    def corpus_profile(self):
        self.profile = {}
        for text in tqdm(self.df.body):
            for w in tokenize([text], self.words_seps, self.chars_to_filter,
                              self.stopwords, self.filter_fun, self.apply_fun):
                self.profile[w] = self.profile[w]+1 if w in self.profile else 1
        self.profile = OrderedDict(sorted(self.profile.items(), key=lambda kv: kv[1], reverse=True))

    def tokenize(self, by='ID'):
        self.tokens = self.df.groupby(by).apply(lambda d: tokenize(d.body.values, self.words_seps, self.chars_to_filter,
                                                                   self.stopwords, self.filter_fun, self.apply_fun))
    
    def set_vocabulary(self):
        if self.profile:
            counter = self.profile
        else:
            all_tokens = [tok for group in self.tokens for tok in group]
            counter = OrderedDict(sorted(Counter(all_tokens).items(), key=lambda kv: kv[1], reverse=True))
        voc = list(counter.keys())
        if self.max_voc_size:
            print(f"Vocabulary: {self.max_voc_size:d}'th word => {list(counter.values())[self.max_voc_size]:d} occurences.")
            voc = voc[:self.max_voc_size]
        if self.min_voc_occurences:
            i = int(np.sum(np.array(list(counter.values()))>=self.min_voc_occurences))
            print(f"Vocabulary: {self.min_voc_occurences:d} occurences => {i:d} words.")
            voc = voc[:i]
        self.vocabulary = voc

    def set_one_hot_embedding(self):
        if not self.tokens:
            self.set_one_hot_on_the_fly()
            return
        counters = {w: len(self.tokens)*[self.laplace_smooth] for w in self.vocabulary}
        for i,group in enumerate(self.tokens):
            for w in group:
                if w in counters:
                    counters[w][i] += 1
        self.one_hot = pd.DataFrame({**counters})

    def set_one_hot_on_the_fly(self):
        filter_fun = self.filter_fun if self.filter_fun else lambda w: w in self.vocabulary
        counters = {w: len(np.unique(self.df.ID)) * [self.laplace_smooth]
                    for w in self.vocabulary}
        for i,ID in enumerate(tqdm(np.unique(self.df.ID))):
            for w in tokenize(self.df[self.df.ID==ID].body.values,
                              self.words_seps, self.chars_to_filter,
                              self.stopwords, filter_fun, self.apply_fun):
                if w in counters:
                    counters[w][i] += 1
        self.one_hot = pd.DataFrame({**counters})
    
    def save(self, path='Data/parsed_data.pkl'):
        with open(path, 'wb') as f:
            pkl.dump(
                {'meta':self.meta, 'df':self.df, 'tokens':self.tokens,
                 'profile':self.profile, 'vocabulary':self.vocabulary, 'one_hot':self.one_hot},
                f
            )

    def load(self, path='Data/parsed_data.pkl'):
        with open(path,'rb') as f:
            tmp = pkl.load(f)
            self.meta = tmp['meta']
            self.df = tmp['df']
            self.tokens = tmp['tokens']
            self.profile = tmp['profile']
            self.vocabulary = tmp['vocabulary']
            self.one_hot = tmp['one_hot']
    
    # Analysis
    def full_protocol(self, ID=None):
        if ID is None:
            ID = random.choice(np.unique(self.df.ID))
        protocol = self.df[self.df.ID==ID]
        return '\n\n'.join(['________\n'+h+':\n'+b for h,b in zip(protocol.header,protocol.body)])
    
    def show_tokens(self, **kwargs):
        if self.profile:
            show_freqs(self.profile, **kwargs)
        else:
            show_freqs([tok for group in self.tokens for tok in group], **kwargs)

    def one_hot_pca(self):
        # TODO plot PCA of one_hot with 2 components
        warn('PCA display of protocols is not implemented :(')
		

def show_freqs(tokens, vocabulary=None, n=15, logscale=True, token_lab='Token', compact=False, show_tail=True):
    n_figs = 2 + show_tail
    if compact:
        _, axs = plt.subplots(1,n_figs, figsize=(18,5))
    else:
        _, axs = plt.subplots(n_figs,1, figsize=(12,12))
    
    if vocabulary:
        vocabulary = set(vocabulary)
        tokens = [token for token in  tokens if token in vocabulary]
    count = tokens if tokens is dict else Counter(tokens)
    count = OrderedDict(sorted(count.items(), key=lambda kv: kv[1]))
    vals = list(count.values())
    keys = list(count.keys())
    
    # quantile plot
    ax = axs[0]
    ax.axhline(np.sum(vals)/len(vals), linestyle=':', color='blue', label='Average')
    ax.plot(list(range(101)), [vals[int(q/100*(len(vals)-1))] for q in range(101)], 'k.-')
    ax.set_xlabel('Quantile [%]')
    ax.set_ylabel('Number of occurences')
    ax.set_xlim((0,100))
    ax.set_title(f'Total: {np.sum(vals):.0f}, Unique: {len(keys):d}')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()
    
    # top tokens
    ax = axs[1]
    ax.bar([bidi.get_display(token) for token in keys[-1:-n-1:-1]], vals[-1:-n-1:-1])
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
    if show_tail:
        ax = axs[2]
        ax.bar([bidi.get_display(token) for token in list(reversed(keys[:n]))], list(reversed(vals[:n])))
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
