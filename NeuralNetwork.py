# -*- coding: utf-8 -*-
"""
Heemani Brahmbhatt
CS 421 - Natural Language Processing
Movie Reviews Sentiment Analysis Neural Network

Description: The following code creates a feedforward neural network to determine whether a movie review is positive or negative, 
and it classifies the review as such. We use metrics such as Accuracy, Precision, Recall, and F1-score to evaluate the classifier.

Attribution Information: The skeleton code for this project was provided by Professor Natalie Parde in a CS as a part of an assignment 
in the course CS 421 - Natural Language Processing, Fall 2020, University of Illinois at Chicago (UIC)
"""

# Import modules
import tensorflow as tf
import numpy as np
import random
from utils import *
import itertools as it



# Function to get average vector from list of vectors
#
# Arguments: A list of vectors
#
# Returns: a single vector of the same dimensions as the vectors in the input list
def avg_vec(vectors):
    length = len(vectors[0])
    size = len(vectors)
    return [np.mean([x[i] for x in vectors]) for i in range(length)]



# Function to get word2vec representations
#
# Arguments:
# reviews: A list of strings, each string represents a review
#
# Returns: mat (numpy.ndarray) of size (len(reviews), dim)
# mat is a two-dimensional numpy array containing vector representation for ith review (in input list reviews) in ith row
# dim represents the dimensions of word vectors, here dim = 300 for Google News pre-trained vectors
def w2v_rep(reviews):
    dim = 300
    mat = np.zeros((len(reviews), dim))
    # Load the pre-trained word vectors
    w2v = load_w2v()
    
    i = 0
    # For every review...
    for review in reviews:
        tokens = get_tokens(review) # split review into tokens
        
        valid_tokens = []
        for x in tokens:
            if x in w2v:
                valid_tokens.append(x) # add the token, only if it's valid
        vecs = []
        for v in valid_tokens:
            vecs.append(w2v[v])
        if not vecs:
            avg_vector = np.zeros(dim)
        else:
            avg_vector = avg_vec(vecs)
        
        mat[i] = avg_vector
        i += 1
    return mat



# Function to build a feed-forward neural network using tf.keras.Sequential model. Build the sequential model
# by stacking up dense layers such that each hidden layer has 'relu' activation. Add an output dense layer in the end
# containing 1 unit, with 'sigmoid' activation, this is to ensure that we get label probability as output
#
# Arguments:
# params (dict): A dictionary containing the following parameter data:
#                    layers (int): Number of dense layers in the neural network
#                    units (int): Number of units in each dense layer
#                    loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#                    optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#
# Returns:
# model (tf.keras.Sequential), a compiled model created using the specified parameters
def build_nn(params):
    num_layers = params['layers']
    num_units = params['units']
    loss_arg = params['loss']
    opt_arg = params['optimizer']
    model = tf.keras.Sequential()
    
    # Add [num_layers] layers to the model, each with [num_units] units and relu activation
    for x in range(num_layers):
        model.add(tf.keras.layers.Dense(num_units, activation='relu')) # Add specified amt of dense layers with relu activation
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # Add last layer with 1 unit and sigmoid activation
    
    model.compile(optimizer=opt_arg, loss=loss_arg) # compile model with loss and optimizer
    
    return model



# Function to compute accuracy metric
#
# Arguments: predictions: python list of the predicted values by model
#            actual: python list of the actual values/labels in the test set
#
# Returns: accuracy: float
def get_acc(predictions, actual):
    length = len(predictions)
    
    accurate = 0
    for x in range(length):
        if predictions[x] == actual[x]:
            #print("IS CORRECT")
            accurate+=1
    accuracy = accurate/length
    
    return accuracy



# Function to select the best parameter combination based on accuracy by evaluating all parameter combinations
# This function trains on the training set (X_train, y_train) and evluates using the validation set (X_val, y_val)
#
# Arguments:
# params (dict): A dictionary containing parameter combinations to try:
#                    layers (list of int): Each element specifies number of dense layers in the neural network
#                    units (list of int): Each element specifies the number of units in each dense layer
#                    loss (list of string): Each element specifies the type of loss to optimize ('binary_crossentropy' or 'mse)
#                    optimizer (list of string): Each element specifies the type of optimizer to use while training ('sgd' or 'adam')
#                    epochs (list of int): Each element specifies the number of iterations over the training set
# X_train (numpy.ndarray): A matrix containing w2v representations for training set of shape (len(reviews), dim)
# y_train (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_train of shape (X_train.shape[0], )
# X_val (numpy.ndarray): A matrix containing w2v representations for validation set of shape (len(reviews), dim)
# y_val (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_val of shape (X_val.shape[0], )
#
# Returns:
# best_params (dict): A dictionary containing the best parameter combination:
#                        layers (int): Number of dense layers in the neural network
#                          units (int): Number of units in each dense layer
#                         loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#                        optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#                        epochs (int): Number of iterations over the training set
def find_best_params(params, X_train, y_train, X_val, y_val):
    
    # Note that you don't necessarily have to use this loop structure for your experiments
    # However, you must call reset_seeds() right before you call build_nn for every parameter combination
    # Also, make sure to call reset_seeds right before every model.fit call

    # Get all parameter combinations (a list of dicts) keep the key, value pairings in permutation
    """# Code Attribution: https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python"""
    keys, values = zip(*params.items())
    param_combinations = [dict(zip(keys, v)) for v in it.product(*values)] # a list of dictionaries holding unique combinations of parameters

    # Iterate over all combinations using one or more loops
    i = 0
    best_accuracy = 0
    best_params = 0
    for param_combination in param_combinations:
        # Reset seeds and build your model
        reset_seeds()
        model = build_nn(param_combination)
        # Train and evaluate your model, make sure you call reset_seeds before every model.fit call
        e = param_combination['epochs']
        reset_seeds()
        model.fit(X_train, y_train, epochs=e)
        predictions = model.predict(X_val)
        preds = []
        for p in predictions:
            if p > 0.5000:
                preds.append(1)
            else:
                preds.append(0)
        local_accuracy = get_acc(preds, y_val)
        if local_accuracy > best_accuracy: # finding the parameter yielding better accuracy
            best_accuracy = local_accuracy
            best_params = i
        i+=1
        
    return param_combinations[best_params]


# Quick function that takes a python list of values (pos, neg) and returns a numpy ndarray where pos=1, neg=0
def to_nums(a_list):
    new_list = []
    for x in a_list:
        if x == "pos":
            new_list.append(1)
        elif x == "neg":
            new_list.append(0)
            
    res = np.asarray(new_list) # convert it into an numpy array, so all pos = 1 and all neg = 0
    
    return res
    
    

# Function to get accuracy, precision, recall, and f1-measure
#
# Arguments: predictions: python list of predicted values
#            actuals: python list of actual values/labels in the test set
# Note: the positive class is the classification of a movie review being negative (0)
def get_metrics(predictions, actuals):
    length = len(predictions)
    preds = [int(x) for x in predictions]
    print(preds)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x in range(length):
        if preds[x] == 0 and actuals[x] == 0:
            tp+=1
        if preds[x] == 0 and actuals[x] == 1:
            fp+=1
        if preds[x] == 1 and actuals[x] == 0:
            fn+=1
        if preds[x] == 1 and actuals[x] == 1:
            tn+=1
        
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)
    
    return accuracy, precision, recall, f1_score



# Function to convert probabilities into pos/neg labels
#
# Arguments:
# probs (numpy.ndarray): A numpy vector containing probability of being positive
#
# Returns:
# pred (numpy.ndarray): A numpy vector containing pos/neg labels such that ith value in probs is mapped to ith value in pred
#                         A value is mapped to pos label if it is >=0.5, neg otherwise
def translate_probs(probs):
    labels = []
    for p in probs:
        if p >= 0.5000:
            labels.append('pos')
        else:
            labels.append('neg')
    my_labels = np.asarray(labels)
    return my_labels
    


# Function to convert the probabilities into a binary classification (0 or 1) numerical
#
# Arguments: probs: the python list of proabbilities predicted by the model
#
# Returns: pred: python list containing 1's and 0's based on the probabilistic predictions
def name_to_num(probs):
  pred = np.repeat('pos', probs.shape[0])
  i = 0
  for p in probs:
    if p >= 0.5000:
      pred[i] = 1
    else:
      pred[i] = 0
    i+=1
  return pred

def main():
    # Load dataset
    data = load_data('movie_reviews.csv')

    # Extract list of reviews from the training set
    # Note that since data is already sorted by review IDs, you do not need to sort it again for a subset
    train_data = list(filter(lambda x: x['split'] == 'train', data))
    reviews_train = [r['text'] for r in train_data]
    
    # Compute the word2vec representation for training set
    X_train = w2v_rep(reviews_train)
    
    # Save these representations in q1-train-rep.npy
    np.save('q1-train-rep.npy', X_train)

    # Extract the rows of the csv that are a part of the validation data
    validation_data = list(filter(lambda x: x['split'] == 'val', data))
    reviews_validate = [r['text'] for r in validation_data]

    X_val = w2v_rep(reviews_validate)

    # Extract test data rows
    test_data = list(filter(lambda x: x['split'] == 'test', data))
    reviews_test = [r['text'] for r in test_data]

    X_test = w2v_rep(reviews_test)

    val_labels = [x['label'] for x in validation_data]
    y_val = to_nums(val_labels)

    #print(y_val)
    
    # Training data
    train_labels = [x['label'] for x in train_data]
    y_train = to_nums(train_labels)

    # Build a feed forward neural network model with build_nn function
    params = {
        'layers': 1,
        'units': 8,
        'loss': 'binary_crossentropy',
        'optimizer': 'adam'
    }
    reset_seeds()
    model = build_nn(params)
    
    # Function to choose best parameters
    # Use build_nn function in find_best_params function
    params = {
        'layers': [1, 3],
        'units': [8, 16, 32],
        'loss': ['binary_crossentropy', 'mse'],
        'optimizer': ['sgd', 'adam'],
        'epochs': [1, 5, 10]
    }

    best_params = find_best_params(params, X_train, y_train, X_val, y_val)

    print("Best parameters: {0}".format(best_params)) # these are the best parameters to train model
    
    
    # Build a model with best parameters and fit on the training set
    # reset_seeds function must be called immediately before build_nn and model.fit function
    reset_seeds()
    model = build_nn(best_params)
    reset_seeds()
    model.fit(X_train, y_train, epochs=best_params['epochs'])

    # Use the model to predict labels for the validation set
    pred = model.predict(X_val).flatten()

    preds = name_to_num(pred)

    #print(preds.shape)

    acc, pre, rec, f1 = get_metrics(preds, y_val) # get all of the metrics

    print("Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(acc, pre, rec, f1))

    # Use the model to predict labels for the test set
    pred = model.predict(X_test)
    
    # Translate predicted probabilities into pos/neg labels
    pred = translate_probs(pred)

    #print(pred.shape)
    
    # Save the results in numpy file
    np.save('q1-pred.npy', pred)

if __name__ == '__main__':
    main()
