from featurizer import get_tweets, get_feats_and_labels, tfidf, bag_of_words
from models import PCA, knn, random_forest

tweets = get_tweets()
data = get_feats_and_labels(tweets)
features = tfidf(data[0])
bin_labels = data[1]
cont_labels = data[2]

print("num of features: ", len(features))

# PCA(features, bin_labels)
print("score: ", knn(features, bin_labels))
# print("score: ", random_forest(features, bin_labels))