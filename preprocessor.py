import re

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from autocorrect import spell

class Preprocessor():
    @staticmethod
    def text_process(tweet):
        stop_words = stopwords.words('english')
        stemmer = WordNetLemmatizer()
        preprocessed_tweets = []

        # Remove Punctuation
        tweet = re.sub(r'\W+', ' ', tweet)

        # Lowercase
        tweet = tweet.lower()

        # Tokenize
        tokenized_tweet = word_tokenize(tweet)

        # Autocorrect
    #         for i in range(len(tokenized_tweet)):
    #             tokenized_tweet[i] = spell(tokenized_tweet[i])

        # Remove stopwords
        # There are stop words that actually help identify 
        # incidents, especially ones that indicates location.
        # In some cases accuracy is better without removing
        # stop words. 
    #         for word in tokenized_tweet:
    #             if word in stop_words:
    #                 tokenized_tweet.remove(word)

        # Stemming
    #         for i in range(len(tokenized_tweet)):
    #             tokenized_tweet[i] = stemmer.lemmatize(tokenized_tweet[i])

        # Add to list
        return tokenized_tweet
    
    @staticmethod
    def processSentences(sentences):
        processedSentences = list()
        i = -1
        for word in sentences:
            if type(word[0]) is str:
                processedSentences.append(list())
                i += 1
            corpus = word[1], word[2], word[3]
            processedSentences[i].append(tuple(corpus))
        return processedSentences