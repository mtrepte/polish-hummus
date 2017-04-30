{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "import random\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_tweets():\n",
    "    tweets = pd.read_csv(\"tweets.csv\", encoding = \"ISO-8859-1\")\n",
    "    return tweets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_feats_and_labels(tweets):\n",
    "    # tweet texts\n",
    "    tweets.feats = tweets['tweet_text']\n",
    "    \n",
    "    # binary hate speech classification (1 or 0)\n",
    "    tweets.bin_labels = tweets['does_this_tweet_contain_hate_speech']\n",
    "    tweets.bin_labels = [1 if label == 'The tweet contains hate speech' else 0 for label in tweets.bin_labels]\n",
    "    \n",
    "    # continuous hate speech metric (0 to 1)\n",
    "    tweets.cont_labels = tweets['does_this_tweet_contain_hate_speech:confidence']\n",
    "    \n",
    "    return [tweets.feats, tweets.bin_labels, tweets.cont_labels]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def bag_of_words(tweets):\n",
    "    random.shuffle(tweets)\n",
    "    featurizer = CountVectorizer()\n",
    "    X = featurizer.fit_transform(tweets)\n",
    "    X = X.toarray()\n",
    "    return X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def tfidf(tweets):\n",
    "    random.shuffle(tweets)\n",
    "    featurizer = TfidfVectorizer()\n",
    "    X = featurizer.fit_transform(tweets)\n",
    "    X = X.toarray()\n",
    "    return X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# example\n",
    "\n",
    "tweets = get_tweets()\n",
    "data = get_feats_and_labels(tweets)\n",
    "features = data[0]\n",
    "binary_labels = data[1]\n",
    "continuous_labels = data[2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
