# The following file utilizes Word2Vec and Google Translate in order to vectorize committees from .csv files. 
# Dependencies: Python 3 (obviously), gensim, googletrans (!pip install ...)

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
import numpy as np
import time
from nltk.corpus import stopwords
from googletrans import Translator
from nltk import word_tokenize
from sklearn.metrics import mean_squared_error

#---~-~--- Now to load the model ---~-~---#
!wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
!pip install nltk
from gensim.models import KeyedVectors 

model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

translator = Translator()
stops = set(stopwords.words('english'))

# Words that appear as .csv headers but are not a Hak speaking
FILTER_OUT = ["נכחו", "חברי הוועדה", "מוזמנים", "סדר היום"]

# Characters to filter out before all the tokenizing process.
CHARS_TO_FILTER = '\t|\n|\r|\"|\'|,|–|;'

# The shape of the vectorized form of our data.
VECTORIZED_SHAPE = 300

# The threshold on the length of the message for us to include the message in the vectorization - used to filter spam.
LENGTH_THRESHOLD = 30

# The maximum amount of lines in the csv to look for the end of the header.
MAX_HEADER_SEARCH = 8

# The maxinum amount of vectorized sentences we allow in a committee vectorization

COMMITTEE_MAX_ALLOWED_SENTENCES = 30

def translate_hebrew_sentence(sentence):
	"""
	Translates a single sentence (text) fro Hebrew to English via Google Translate.
	:param sentence: The text to translate
	:return: the translated text.
	"""
	return translator.translate(sentence).text


def tokenize_hebrew_sentence(sentence):
	"""
	Translates a single sentence (text) fro Hebrew to English via Google Translate.
	In addition, the English data is then toeknized and filtered.
	:param sentence: The original text
	:return: the translted text.
	"""

	text = translate_hebrew_sentence(sentence)
	tokenized_sentence = word_tokenize(text)
	filtered = []
	

	# Replace stop charaters/words.
	for i in tokenized_sentence:
		i = re.sub(CHARS_TO_FILTER, '', i)

		if len(i) < 1:
			continue

		if i in stops:
			continue
		
		filtered.append(i)
		
	return filtered



def vectorize_tokenized_sentence(filtered_tokenized_sentence):
	"""
	Vectorizes the given sentence, if possibe. This is done via scanning the tokens in the sentence and
	vectorizing each of them.
	:param filtered_tokenized_sentence: The tokenized data to work on.
	:return: The vector form of the sentence.
	"""
	vectorized = np.zeros(VECTORIZED_SHAPE)
	vsize = 0

	for i in filtered_tokenized_sentence:
		if model.vocab.has_key(i):
			vectorized += model[i]
			vsize += 1

	result = np.array(filtered)

	if vsize > 0:
		vectorized = vectorized / vsize
	
	return vectorized


def vectorize_committee(df):
	"""
	Vectorizes an entire committee. It looks for longer parts in the protocol that are not headers,
	then vectorizes the sentences in there.
	:param df: the dataFrame of the protocol (.csv file)
	"""
	i = 2 # Works like contant so far
	while i < MAX_HEADER_SEARCH:
		try:
			if df.iloc[i].values[0] not in FILTER_OUT:
				break
		except:
			pass
		
		i += 1
		
	df = df.iloc[i:]
	interesting_parts = df[~pd.isnull(df["body"])] 
	interesting_parts = interesting_parts[interesting_parts["body"].apply(len) >= LENGTH_THRESHOLD]

	vector = np.zeros(VECTORIZED_SHAPE)
	vector_size = 0
	
	i = 0 
	for p in np.array(interesting_parts["body"].values):
		i += 1
		if i > COMMITTEE_MAX_ALLOWED_SENTENCES:
			break
		
		vector += vectorize_tokenized_sentence(tokenize_hebrew_sentence(p))
		vector_size += 1
	
	return vector / vector_size    