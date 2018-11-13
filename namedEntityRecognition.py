import nltk
import pandas as pd
import pickle as pkl
import sklearn
import scipy.stats
import sklearn_crfsuite

from itertools import chain
from sklearn.metrics import make_scorer
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

def readCSV(file):
    csv = pd.read_csv(file, encoding = "ISO-8859-1")
    return csv

def readTweetForTesting(file):
    csv = pd.read_csv(file, sep="\t")
    data = pd.DataFrame({'tweets':csv['tweets'], 'isRoadIncident':csv['isRoadIncident']})[['tweets', 'isRoadIncident']]
    return data

def saveModel(model, file):
    output = open('models/%s' % file, 'wb')
    pkl.dump(model, output)
    output.close()

def loadModel(file):
    input = open('models/%s' % file, 'rb')
    data = pkl.load(input)
    return data

def extractFeatureFromWords(sentence, idx):
    word = sentence[idx][0]
    postag = sentence[idx][1]

    features = {
        'bias': 1.0,
        'lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'isSupper': word.isupper(),
        'isTitle': word.istitle(),
        'isDigit': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if idx > 0:
        word1 = sentence[idx-1][0]
        postag1 = sentence[idx-1][1]
        features.update({
            '-1:lower': word1.lower(),
            '-1:isTitle': word1.istitle(),
            '-1:isSuper': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if idx < len(sentence)-1:
        word1 = sentence[idx+1][0]
        postag1 = sentence[idx+1][1]
        features.update({
            '+1:lower': word1.lower(),
            '+1:isTitle': word1.istitle(),
            '+1:isSuper': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def extractFeatureFromSentence(sent):
    return [extractFeatureFromWords(sent, i) for i in range(len(sent))]

def labelsExtraction(sent):
    return [label for token, postag, label in sent]

def tokenize(sent):
    return [token for token, postag, label in sent]

def featureExtraction(data):
    # Even data splitting
    train_sentences = data[:43164]
    test_sentences = data[43164:47959]

    X_train = [extractFeatureFromSentence(sentence) for sentence in train_sentences]
    y_train = [labelsExtraction(sentence) for sentence in train_sentences]

    X_test = [extractFeatureFromSentence(sentence) for sentence in test_sentences]
    y_test = [labelsExtraction(sentence) for sentence in test_sentences]

    return (X_train, y_train, X_test, y_test)

def postag(tweet):
    sent_text = nltk.sent_tokenize(tweet)
    sent = list()
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized_text)
        sent.append(tagged)
    return sent

def locationExtraction(tagged, y):
    locations = []
    for i in range (0, len(y)):
        loc = ''
        for j in range(0, len(y[i])):
            if y[i][j] == 'B-geo':
                if len(loc) == 0:
                    loc += tagged[i][j][0]
                else:
                    locations.append(loc)
                    loc = tagged[i][j][0]
            if y[i][j] == 'I-geo':
                loc += ' ' + tagged[i][j][0]
        if len(loc) > 0:
            locations.append(loc)
    return locations

def predictLocation(tweets):
    y_pred = pipeline.predict(tweets)
    
    locations = []
    for i in range(0, len(tweets)):
        if y_pred[i] == 1:
            tagged = postag(tweets[i])
            x = [extractFeatureFromSentence(s) for s in tagged]
            y = crf.predict(x)
            y_loc = locationExtraction(tagged,y)
            if len(y_loc) > 0:
                locations.append(y_loc) 
            else:
                locations.append('-')
        else:
            locations.append('-')
            
    df = pd.DataFrame({'tweets':tweets, 'isRoadIncident':y_pred, 'location':locations})[['tweets', 'isRoadIncident', 'location']]
    
    return df

class LocationPredictor():
    crf = None

    def __init__(self, file):
        self.crf = loadModel(file)
    
    @staticmethod
    def extractFeatureFromWords(sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'lower': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'isSupper': word.isupper(),
            'isTitle': word.istitle(),
            'isDigit': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:lower': word1.lower(),
                '-1:isTitle': word1.istitle(),
                '-1:isSuper': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:lower': word1.lower(),
                '+1:isTitle': word1.istitle(),
                '+1:isSuper': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    @staticmethod
    def extractFeatureFromSentence(sent):
        return [extractFeatureFromWords(sent, i) for i in range(len(sent))]

    @staticmethod
    def postag(tweet):
        sent_text = nltk.sent_tokenize(tweet)
        sent = list()
        for sentence in sent_text:
            tokenized_text = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokenized_text)
            sent.append(tagged)
        return sent

    @staticmethod
    def locationExtraction(tagged, y):
        locations = []
        for i in range (0, len(y)):
            loc = ''
            for j in range(0, len(y[i])):
                if y[i][j] == 'B-geo':
                    if len(loc) == 0:
                        loc += tagged[i][j][0]
                    else:
                        locations.append(loc)
                        loc = tagged[i][j][0]
                if y[i][j] == 'I-geo':
                    loc += ' ' + tagged[i][j][0]
            if len(loc) > 0:
                locations.append(loc)
        return locations

    @staticmethod
    def predictLocation(model, tweets):        
        locations = []
        for i in range(0, len(tweets)):
            tagged = postag(tweets[i])
            x = [extractFeatureFromSentence(s) for s in tagged]
            y = model.predict(x)
            y_loc = locationExtraction(tagged,y)
            if len(y_loc) > 0:
                locations.append(y_loc) 
            else:
                locations.append('-')
        
        return locations

if __name__ == '__main__':
    preprocessor = Preprocessor()

    # Load data
    # Source: Kaggle
    # https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
    trainingFile = 'datasets/ner-dataset.csv'
    trainCSV = readCSV(trainingFile)

    sentences = preprocessor.processSentences(trainCSV.values)

    X_train, y_train, X_test, y_test = featureExtraction(sentences)

    # Training
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)
    # print(y_pred)
    print(metrics.flat_f1_score(y_test, 
                          y_pred,
                          average='weighted', 
                          labels=['B-geo', 'I-geo']))
    
    tweetsFile = 'datasets/tweet-dataset.csv'
    tweetsTestData = readTweetForTesting(tweetsFile)['tweets'].head(50)
    # pipelineModelFile = 'multinomialNB.pkl'
    # pipeline = loadModel(pipelineModelFile)
    # result = predictLocation(tweetsTestData)
    # print(result)

    # Save Model
    # saveModel(crf, 'namedEntityRecognition.pkl')



