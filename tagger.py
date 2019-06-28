#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The following file uses Yap in order to lemmatize Hebrew sentence through the web service (./yap api)
# Dependencies: Yap on Python 3 (obviously).
# For more information regarding Yap, please check out: https://github.com/onlplab/yap

import os 
import json
import re
from datetime import datetime


FIELDS = [
	(2, "original"),
	(3, "lemma"),
	(4, "original_pos"),
	(5, "lemma_pos"),
	(6, "token_serial")
]

ORIGINAL_FIELD = 3
LEMMA_FIELD = 3
ORIGINAL_POS = 4
LEMMA_POS = 5

# Characters to filter out before all the tokenizing process.
CHARS_TO_FILTER = '\t|\n|\r|\"|\'|,|–|;'

# The 'interesting' parts of speech Yap has to offer that we can extract from our model.
FILTERED_PART_OF_SPEECH = ["JJ", "CD", "NN", "NNT", "NNP", "VB"]

def fetch_md_from_rest(text):
	"""
	Fetches the md_lattice field from Yap's curl service.
	:param text: the text to sed to Yap
	:return: the raw, JSON response.
	"""

	s = 'curl -s -X GET -H \'Content-Type: application/json\' -d\'{\"text\": \"' + text + '  \"}\' localhost:8000/yap/heb/joint | jq .'
	return os.popen(s).read()	


def get_stemmed_sentence(sentence):
	"""
	Stems a sentence via simple analysis.
	:param sentence: the original sentence.
	:return: the stemmed version.
	"""
	sentence = sentence.strip()
	sentence = re.sub(CHARS_TO_FILTER, ' ', sentence)
	return sentence


def split_sentences(text):
	"""
	Splits a given text to a list of stemmed sentences. This is because Yap can only analyze a single chunk of data at a time.
	:param text: the text to stemm.
	:return: The stemmed version of some of the messages.
	"""
	return [get_stemmed_sentence(i) for i in text.split(".")]


def tag_sentence(text, final):
	"""
	Tags a sentence to a dictionary given. Saves the result of Yap's analyis, filtered to the Parts of Speech we introduced.
	:param text: the text to tag (unstemmed sentence).
	:return: None, but the dictionary is now processed.
	"""
	text = get_stemmed_sentence(text)
	raw_parts_os_speech = []
	md_result = json.loads(fetch_md_from_rest(text))
	md_result = md_result["md_lattice"]	
	raw_parts_os_speech = md_result.strip().split("\n")

	for part_of_speech in raw_parts_os_speech:
		fields = part_of_speech.split("\t")
		if len(fields) < 3:
			continue

		if fields[LEMMA_POS] in FILTERED_PART_OF_SPEECH:
			if fields[LEMMA_FIELD] != "_":
				final[fields[LEMMA_POS]].append(fields[LEMMA_FIELD])
			else:
				final[fields[LEMMA_POS]].append(fields[ORIGINAL_FIELD])


def lemmatize(raw):
	"""
	Lemmatizes the raw, given data, via tagging each sentence and summing using tag_sentence.
	:param: raw the data to analyze.
	:return: the lemmatization dictionary: appearances of the parts  of speech mentioned.
	"""
	final = { pos: [] for pos in FILTERED_PART_OF_SPEECH }
	if not raw:
		return final
	for s in split_sentences(raw):
		if not s:
			continue
		tag_sentence(s, final)
	return final	

print(lemmatize("המאבק בתאונות הדרכים - אכיפת עבירות התנועה"))
	