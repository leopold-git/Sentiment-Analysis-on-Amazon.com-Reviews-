import nltk
import os
import random
from nltk.tokenize import word_tokenize
import collections
from nltk.metrics import (precision, recall, f_measure)


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


# function that establishes whether words are contained in a given doc
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


#train_set, test_set = featuresets[400:], featuresets[:400]

### cross val split occurs here  -- 10 folds
num_folds = 10
subset_size = int(round(len(featuresets)/num_folds))

foldAccuracies = []
foldNegativePrecisions = []
foldNegativeRecalls = []
foldNegativeFScores = []
foldPositivePrecisions = []
foldPositiveRecalls = []
foldPositiveFScores = []

for i in range(num_folds):
    cv_test = featuresets[i*subset_size:][:subset_size]
    cv_train = featuresets[:i*subset_size] + featuresets[(i+1)*subset_size:]
    # use NB classifier
    classifier = nltk.NaiveBayesClassifier.train(cv_train)
    print('  ')
    print('FOLD ' + str(i))
    print('For this fold:')
    print('Accuracy on Fold Test Set: ' + str(nltk.classify.accuracy(classifier, cv_test)))
    foldAccuracies.append(str(nltk.classify.accuracy(classifier, cv_test)))
    #most informative feauures
    classifier.show_most_informative_features(10)  # get 5 most informative feats
    # now get fold stats such as precison, recall, f score
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(cv_test):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    foldPositivePrecisions.append(str(precision(refsets['positive'], testsets['positive'])))
    foldPositiveRecalls.append(str(recall(refsets['positive'], testsets['positive'])))
    foldPositiveFScores.append(str(f_measure(refsets['positive'], testsets['positive'])))
    foldNegativePrecisions.append(str(precision(refsets['negative'], testsets['negative'])))
    foldNegativeRecalls.append(str(recall(refsets['negative'], testsets['negative'])))
    foldNegativeFScores.append(str(f_measure(refsets['negative'], testsets['negative'])))



    print('Positive Precision:', precision(refsets['positive'], testsets['positive']))
    print('Positive Recall:', recall(refsets['positive'], testsets['positive']))
    print('Positive F1-Score:', f_measure(refsets['positive'], testsets['positive']))
    print('Negative Precision:', precision(refsets['negative'], testsets['negative']))
    print('Negative Recall:', recall(refsets['negative'], testsets['negative']))
    print('Negative F1-Score:', f_measure(refsets['negative'], testsets['negative']))

    # train using training_this_round
    # evaluate against testing_this_round
    # save accuracy
total = 0
totalPrecPos = 0
totalRecallPos = 0
totalFScorePos = 0
totalPrecNeg = 0
totalRecallNeg = 0
totalFScoreNeg = 0
for i in range(0, len(foldAccuracies)):
    total = total + float(foldAccuracies[i])
    totalPrecPos = totalPrecPos + float(foldPositivePrecisions[i])
    totalRecallPos = totalRecallPos + float(foldPositiveRecalls[i])
    totalFScorePos = totalFScorePos + float(foldPositiveFScores[i])
    totalPrecNeg = totalPrecNeg + float(foldNegativePrecisions[i])
    totalRecallNeg = totalRecallNeg + float(foldNegativeRecalls[i])
    totalFScoreNeg = totalFScoreNeg + float(foldNegativeFScores[i])

total_accuracy = total/num_folds
total_pos_prec = totalPrecPos/num_folds
total_pos_recall = totalRecallPos/num_folds
total_pos_fscore = totalFScorePos/num_folds
total_neg_precision = totalPrecNeg/num_folds
total_neg_recall = totalRecallNeg/num_folds
total_neg_fscore = totalFScoreNeg/num_folds
print('---------')
print('Averaged model performance over 10 folds: ')
print('   ')
print('Average accuracy over 10 folds: ' + str(total_accuracy))
print('Average precision for positive class: ' + str(total_pos_prec))
print('Average recall for positive class ' + str(total_pos_recall))
print('Average F-score for positive class ' + str(total_pos_fscore))
print('  ')
print('Average precision for negative class ' + str(total_neg_precision))
print('Average recall for negative class ' + str(total_neg_recall))
print('Average F-score for negative class ' + str(total_neg_fscore))







