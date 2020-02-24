import nltk
import os
import random
from nltk.tokenize import word_tokenize

directory = 'DC1/neg/'
directory2 = 'DC1/pos/'

# load in all reviews and assign labels to each review, then shuffle order
labeledReviews = ([(os.path.join(directory, filename), 'negative') for filename in os.listdir(directory)] + [(os.path.join(directory2, filename), 'positive') for filename in os.listdir(directory2)])
random.shuffle(labeledReviews)

listOfWords = []
text = ""



# open and read all files--append to one large text to later tokenize
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    f = open(filepath, 'r', encoding='latin-1')
    text += f.read()
   # tokens = word_tokenize(f.read())
   # labeledReviews.append((tokens, 'negative'))

for filename in os.listdir(directory2):
    #print(filename)
    filepath = os.path.join(directory2, filename)
    f = open(filepath, 'r', encoding='latin-1')
    text += f.read()
    #tokens = word_tokenize(f.read())
   # labeledReviews.append((tokens, 'positive'))

# take text containing all words and tokenize it
tokens =  nltk.word_tokenize(text)
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]

# get frequency dist of all words
all_words = nltk.FreqDist(words)
#print(all_words)
#take 2000 most prominent words
word_features = list(all_words)[:2000]
#print((word_features))

def document_features(document):
    document_words = set(document)
    #print(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# need to open the files and read at every labeled review
#featuresets = [(document_features(d), c) for (d,c) in labeledReviews]
# loop and append to featuresets
featuresets = []
for (d,c) in labeledReviews:
   # print(d,c)
    f = open(d, 'r', encoding='latin-1')
    docToken = word_tokenize(f.read())
    feat = document_features(docToken)
    featuresets.append((feat, c))
  ## problem


train_set, test_set = featuresets[400:], featuresets[:400]


# regression model

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

# use NB classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Accuracy on Test Set: ' + str(nltk.classify.accuracy(classifier, test_set)))
classifier.show_most_informative_features(20) # get 5 most informative feats


# precision, recall, F-score


# logistic regression model
import collections
from nltk.metrics import (precision, recall, f_measure)

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)





for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print('Positive Precision:', precision(refsets['positive'], testsets['positive']))
print('Positive Recall:', recall(refsets['positive'], testsets['positive']))
print('Positive F1-Score:', f_measure(refsets['positive'], testsets['positive']))
print('Negative Precision:', precision(refsets['negative'], testsets['negative']))
print('Negative Recall:', recall(refsets['negative'], testsets['negative']))
print('Negative F1-Score:', f_measure(refsets['negative'], testsets['negative']))

