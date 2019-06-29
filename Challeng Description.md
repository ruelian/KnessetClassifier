# KnessetClassifier
Classifier of Knesset committees protocols by topic.


## Basic TDL

Legend: **important**, *totally optional*

1. Data and API:
    - **get additional info (e.g. names of committees)**
    - *make data loading and parsing more efficient*
    - be part of the pipeline
    - *RestAPI*
    - write documentation and presentation
2. Protocols-oriented research:
    - **remove non-informative words** (especially if using clustering rather than classification) - not trivial since there're millions of words...
    - **weight/only parse "important" sections (e.g. intro & summary)**
    - use other features in addition to words counters (e.g. lengths, english words, numbers, etc.)
3. Apply more advanced NLP tools:
    - detect entities and increase their weight
    - lemmatize [[1](https://github.com/synhershko/HebMorph),[2](https://docs.hebrew-nlp.co.il/#/Morph/Normalize)]
    - use word embedding [[1](https://github.com/NLPH/NLPH_Resources#embeddings),[2](https://github.com/liorshk/wordembedding-hebrew)] (plug-and-play use will apparently require lemmatization)
    - understand deep LDA from [reference](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05) (possibly different embedding); go over [gensim](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)
4. Apply clustering and/or classification:
    - **quickly apply LDA clustering and study the outputs** - are they reasonably useful? are there clear ways to improve them? what does the potential of the method look like, compared to the project goals?
    - **think how it is possible to change from clustering to (supervised) classification** (which is a much easier task since the learning method can learn to focus on the relevant information).
    - consider various methods for clustering/classification
5. Continuously review the status & the outputs and see what's missing to achieve the goals.


## Official tasks
(sorted by awarded points)
1. **Topics classification** (35).
2. **Topics definition** (20): we should discuss with the supervisor what the **expected goals and usages** are, and what **possible sources of topics' lists** may be available in addition to the protocols themselves.
3. **Pipeline interface** (20).
4. **Tracking persons over time** (10): simple stats should be easy to obtain (don't they exist already?) - e.g. percent of attended discussions in each committe, attendence in discussions over time, amount of talking within discussions, etc. In general, we'll probably be able to summarize most outputs in a per-person-by-time way. We should **ask if there're any specific requirements**.
5. **Sub-topics classification** (10): hundreds of sub-topics sound quite impractical. **Without supervised labels, any clustering will surely get lost** in the noise. I believe our best chance to fully achieve this goal is if we manage to think of some way to create a list of labels (maybe there exist such external list in the Knesset or other sources?) and match them to either protocols (for learning) or typical vocabulary. That sounds tough. Another option is to apply **minor sub-clustering within each cluster**. **It may also be possible to create some service of "find protocols which are most similar to"** - which is applicationally similar, but requires only similarity metric rather than actual clusters.
6. **What each attendee does in the discussion** (10): if the discussions will be labeled with several labels, then the texts of each particular attendee can be analyzed to draw a subset of those labels, though it will require quite high resolution of topics and will be noisier due to fewer data per sample and lack of context.
7. **Opinion of attendees** (5): sounds impractical - works reasonably well in English using state of the art RNNs based on strong embeddings, none of which available in Hebrew. **As an inferior task, we can try to cluster attendees within a topic according to their opinions**. one way to do it may be taking the important words (known from the topic clustering algorithm), and clustering the speakers according to the contexts in which they speak those important words.
8. **RestAPI** (5).
