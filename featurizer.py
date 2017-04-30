from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
import pandas as pd

stop_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','can','will','just','should','now']

def get_tweets():
    tweets = pd.read_csv("tweets.csv", encoding = "ISO-8859-1")
    return tweets

def get_feats_and_labels(tweets):
    # tweet texts
    tweets.feats = tweets['tweet_text']
    
    # binary hate speech classification (1 or 0)
    tweets.bin_labels = tweets['does_this_tweet_contain_hate_speech']
    tweets.bin_labels = [1 if label == 'The tweet contains hate speech' else 0 for label in tweets.bin_labels]
    
    # continuous hate speech metric (0 to 1)
    tweets.cont_labels = tweets['does_this_tweet_contain_hate_speech:confidence']
    
    return [tweets.feats, tweets.bin_labels, tweets.cont_labels]

def bag_of_words(tweets):
    # random.shuffle(tweets)
    featurizer = CountVectorizer(stop_words=stop_words)
    X = featurizer.fit_transform(tweets)
    X = X.toarray()
    return X

def tfidf(tweets):
    # random.shuffle(tweets)
    featurizer = TfidfVectorizer(stop_words=stop_words)
    X = featurizer.fit_transform(tweets)
    X = X.toarray()
    return X