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

CHARS_TO_FILTER = '\t|\n|\r|\"|\'|,|–|;'
FILTERED_PART_OF_SPEECH = ["JJ", "CD", "NN", "NNT", "NNP", "VB"]

def fetch_md_from_rest(text):
	s = 'curl -s -X GET -H \'Content-Type: application/json\' -d\'{\"text\": \"' + text + '  \"}\' localhost:8000/yap/heb/joint | jq .'
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
			final[fields[5]].append(fields[3])

		# processed_parts.append(processed_part)


def lemmatize(raw):
	final = { pos: [] for pos in FILTERED_PART_OF_SPEECH }
	for s in split_sentences(raw):
		tag_sentence(s, final)
	return final

print(lemmatize("התכוונתי להתייחס לנקודה אחת לעניין ההערה לסעיף (3) בעניין ההגדרה של הפרדה של פסולת אריזות מפסולת אחרת. אנחנו מכירים את הסוגיה, וניתוח שלנו מעלה שיש צורך לעשות התאמה נקודתית בסעיף 22(א) כשנגיע אליו בין הפרדת פסולת אריזות לבין הזרם היבש כי האריזות הוא חלק מהזרם היבש. יש מקום לעשות תיקון נקודתי. בסעיף 1 שזה סעיף מטרה מוצע להשאיר את זה באופן כללי ולעשות התאמה למונח כללי יותר בסעיף של הפרדה של פסולת אריזות מפסולת אחרת. זה יכול להיות לעניין הפרדה במקור או לעניין של הפרדה ופינוי של פסולות, לרבות פסולת אריזות. פשוט להשאיר את זה במונח כללי בסעיף 1. את ההתאמה הנקודתית לזרם היבש נעשה ב-22(א) כשנגיע לשם. ההערה מקובלת ומוצע באמת להשמיט את הביטוי \"פסולת אריזות מפסולת אחרת\" ולהשאיר מונח כללי של הפרדה במקור."))