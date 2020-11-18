# Movie-Review-Sentiment-Analysis
A Feedforward Neural Network to build a classifier that can reliably label positive movies reviews as positive, and negative movie reviews as negative


The neural network is currently trained on a .csv file that is named "movie_reviews.csv". (contains ~50,000 entries)

The movie_reviews.csv file contains the following fields per entry:
          • id: a unique numerical identifier for the review
          • label ∈ {pos, neg, ∅}: positive/negative label for the review (no labels for the test set)
          • split ∈ {train, val, test}: indicates whether the entry belongs in the train, test or validation split
          • text: the space-separated tokens of the review text
          
The python script uses a word embeddings file w2v.pkl - word embeddings from https://code.google.com/archive/p/word2vec/ 
