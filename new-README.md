# KnessetClassifier
*All the italics are things you should fill in/instructions.*

## Team members
*I put our emails in here in case people have further questions. If you'd rather people not email you, I'll just put my email in here. Let me know :)
Also, I have no idea if I spelled your names right in English, so feel free to fix it if I messed it up.*
- Eliana Ruby (elr6577@gmail.com)
- Ido Greenberg (idogreenberg90@gmail.com)
- Tomer Loterman (lotem.tomer1@gmail.com)
- Noam Bresler (noamzbr@gmail.com)
- Yonatan Schwammenthal (yonatansc97@gmail.com)
- Ofek Zicher (ofek.zicher40@gmail.com)


## What we did
Generally speaking, we threw a bunch of things at the wall to see what stuck. Some things turned out more promising than others, and most of them found at least some interesting results. We're putting a brief explanation of everything we tried and what problems we ran into, in the hopes that someone else might want to continue what we started.

1. Data Exploration - *Ido*
2. Language parsing with yap - *Ofek and Noam*
3. Topic modeling with LDA - *Eliana and Yonatan*
4. Document classification with word2vec and Google Translate - *Ofek, and I have no idea if this is a good title so feel free to change it.*
5. Measure of MK participation by topic - *Yonatan and Noam*

## 1. Data Exploration
*Ido*

...


## 2. Language Parsing (yap)
*Ofek and Noam*

...


## 3. Topic modeling (LDA)
*Schwammi - I filled this one out, but feel free to add.*

LDA is a generative statistical topic model. Here's basically how it works:
    - The model gets a bunch of documents, 
    - Each document is modeled as a collection of words (aka a Bag of Words model).
        Note: this requires some preprocessing to turn documents into lists of words (aka tokens).
    - The model thinks of topics as soft clusters of words - basically, a topic is an object that spits out words with various probabilities. (For example, the topic "Animals" will spit out "dog", "cat", and "zoo" with a high probability, and it will spit out "book", "umbrella", and "parachute" with a lower probability.
    - The model thinks of documents as soft clusters of topics - a document is something that is made up of different topics (for example, a document could be 30% about animals, 50% about food and 20% about German philosophy).
    - Given a collection of documents, the model infers what the topic clusters are by seeing what words come up often in the same documents.
    
For a more in-depth explanation, check out [this article](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28).

In order to use LDA, you'll have to do the following steps:
    1. Preprocessing:
        - tokenization (separating the document into a list of words)
        - stopwords (getting rid of words junk words like "and" or "hi" that show up a lot but aren't indicative of anything)
        - filtering by parts of speech - this is a really good way of getting rid of junk words. using just the nouns or just nouns and verbs usually makes for better models.
        - lemmatization - converting all conjugations (הטיות) of words to a base form. (For example: turning "סטודנטים", "הסטודנט", and "סטודנטית" into "סטודנט". This is also really important for the model, because it combines a bunch of words that are basically duplicates into just one word.
    2. Training the model - after preprocessing all your data, you feed it to the model and see what happens. We used [gensim](https://radimrehurek.com/gensim/), but sklearn also has a good one.
    3. Evaluating the model - after training the model, you can have a look at what came out. We used [pyLDAvis](https://pypi.org/project/pyLDAvis/), which is how you get the awesome visualizations you can see in our notebooks. Other than generally checking to see whether the model makes sense, you can also use gensim to check the topic coherence (we did this in our code). Generally, a topic coherence above 0.6 is considered good (we got to about 0.5).
    4. Inference - after building the model, you can give it documents and see what topics they belong to.
    
For a more in-depth explanation and usage guide, check out [this article](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/).

Problems we ran into:
    - like we mentioned in the part about language parsing, YAP is kinda slow so we couldn't process all our data. You can train the model on a subset of data (which [we tried by only running it on the titles](/blob/master/LDA_title_pipeline.ipynb)), but then it's less good. Also, if you want to use the model for inference on new documents, you have to do the same preprocessing on the new documents so that the model can make sense of them.
    - even without having to preprocess everything, the running time is still annoyingly long. In fact, *not* preprocessing the data makes the problem worse, because then it has to deal with more words which makes the running time go up linearly. (We tried running it on [just the Science and Technology committee](/blob/master/LDA_uncleaned_scitech.ipynb), and got results that were kind of okay?
    
**Bottom line: This could actually work pretty well if we had more time and computing power to work with. Hopefully someone can take our code and use it to do that.**


## 4. Document classification (word2vec)
*Ofek*

...


## 5. Measure of MK participation
*Yonatan and Noam*

...
