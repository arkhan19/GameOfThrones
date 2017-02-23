# Goal : Create Word Vectors from I/P Dataset and Analyze for Similarity

from __future__ import absolute_import, print_function, division
import codecs # for word encoding
import glob #regex
import multiprocessing #Concurrency
import os #dealing with reading of files
import pprint #printing human readable
import re # r e e e e e e
import nltk #nltoook kit
import gensim.models.word2vec as wtv #Word2Vec
import sklearn.manifold # dimensionalty reduction
import numpy as np # Math
import matplotlib.pyplot as plt # Plotting stuff
import pandas as kungfu
import seaborn as sb


#Pre-Prcess Data
nltk.download('punkt')
nltk.download('stopwords')

#finding files
booksie = sorted(glob.glob("GameOfThrones/data/*.txt"))

#Creating corpus
corpus = u"" #unicode string so u

for reader in booksie:
    print("Reading '{0}'...".format(booksie))
    with codecs.open(reader, "r", "utf-8") as book:
        corpus += book.read()
    print('Corpus is now {0} characters long'.format(len(corpus)))
    print()

# Tokenization and removing stopworkds
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw = tokenizer.tokenize(corpus)

def sentoword(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sent in raw:
    if len(raw_sent) > 0:
        sentences.append(sentoword(raw_sent))


token_count = sum([len(sentence) for sentence in sentences])
print("The Book corpus contains {0:,} tokens".format(token_count))

###### After Tokenizations, we get vectors. We can use them to get Distance, similarity or ranking ######


# Training Model w2v
# Dimensions of resulting vector
num_feature = 300 # 300 most words

# Minimum word count threshold (for regonizes)
min_word_count = 3

# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()

# context window length
context_size = 7 # Size of what we are looking at a time

# Downsample for frequent words
downsampling = 1e-3 #

# Seed for RNG, to make the results reproducible (rapid number generator)
seed = 1

# la la land
gotsim = wtv.Word2Vec(
                sg=1,
                seed = seed,
                workers = num_workers,
                size = num_feature,
                min_count = min_word_count,
                window = context_size,
                sample = downsampling
            )

gotsim.build_vocab(sentences)
print("WtV vocab length:", len(thrones2vec.vocab))

# End of La la Land now training begins
gotsim.train(sentences)

# Now save the training model PLEASEEEEEE
if not os.path.exists("trained"):
    os.mkdir("trained")
gotsim.save(os.path.join("GameOfThrones/data/trained", "gotsim.w2v"))


