from featurizer import get_tweets, get_feats_and_labels, tfidf, bag_of_words
from models import PCA

tweets = get_tweets()
data = get_feats_and_labels(tweets)
features = tfidf(data[0])
labels = data[1]

print(len(features))
# features = features[:1000]
# labels = labels[:1000]


PCA(features, labels)