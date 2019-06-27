#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

CHARS_TO_FILTER = '\t|\n|\r|\"|\'|,|–|;|\(|\)|\_'
FILTERED_PART_OF_SPEECH = ["JJ", "CD", "NN", "NNT", "NNP", "VB"]

def fetch_md_from_rest(text):
	s = 'curl -s -X GET -H \'Content-Type: application/json\' -d\'{\"text\": \"' + text + '  \"}\' localhost:8000/yap/heb/joint | ./jq .'
	return os.popen(s).read()	


def get_stemmed_sentence(sentence):
	sentence = sentence.strip()
	sentence = re.sub(CHARS_TO_FILTER, ' ', sentence)
	return sentence



def split_sentences(text):
	return [get_stemmed_sentence(i) for i in text.split(".")]


def tag_sentence(text, final):
	text = get_stemmed_sentence(text)
	raw_parts_os_speech = []
	md_result = json.loads(fetch_md_from_rest(text))
	md_result = md_result["md_lattice"]	
	raw_parts_os_speech = md_result.strip().split("\n")

	# processed_parts = []

	for part_of_speech in raw_parts_os_speech:
		fields = part_of_speech.split("\t")
		if len(fields) < 3:
			continue

		# processed_part = {}

		if fields[5] in FILTERED_PART_OF_SPEECH:
			if fields[3] == "_":
				final[fields[5]].append(fields[2])
			else:
				final[fields[5]].append(fields[3])

		# processed_parts.append(processed_part)


def lemmatize(raw):
	final = { pos: [] for pos in FILTERED_PART_OF_SPEECH }
	if not raw:
		return final
	for s in split_sentences(raw):
		if not s:
			continue
		tag_sentence(s, final)
	return final
