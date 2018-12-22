# Fortune Cookie Classification using Naive Bayes

## Description

You will build a binary fortune cookie classifier. This classifier will be used to classify fortune cookie messages into two classes: messages that predict what will happen in the future (class 1) and messages that just contain a wise saying (class 0). 

For example:

"Never go in against a Sicilian when death is on the line" would be a message in class 0.
"You will get an A in Machine learning class" would be a message in class 1.

## Files Provided 

There are three sets offiles. All words in thesefiles are lower case and
punctuation has been removed.

1) The training data:

traindata.txt: This is the training data consisting of fortune cookie messages.

trainlabels.txt: Thisfile contains the class labels for the training data.

2) The testing data:

testdata.txt: This is the testing data consisting of fortune cookie messages.

testlabels.txt: Thisfile contains the class labels for the testing data.

3) A list of stopwords: stoplist.txt

## Steps

In the pre-processing step, you will convert fortune cookie messages into features to be used by your classifier. You will be using a bag of words representation. The following steps outline the process involved: Form the vocabulary. The vocabulary consists of the set of all the words that are in the training data with stop words removed (stop words are common, uninformative words such as "a" and "the" that are listed in thefile stoplist.txt). The vocabulary will now be the features of your training data. Keep the vocabulary in alphabetical order to help you with debugging. Now, convert the training data into a set of features. Let M be the size of your vocabulary. For each fortune cookie message, you will convert it into a feature vector of size M. Each slot in that feature vector takes the value of 0 or 1. For these M slots, if the ith slot is 1, it means that the ith word in the vocabulary is present in the fortune cookie message; otherwise, if it is 0, then the ith word is not present in the message. Most of these feature vector slots will be 0. Since you are keeping the vocabulary in alphabetical order, thefirst feature will be the first word alphabetically in the vocabulary.

## Implemenation

a) Implement the Naive Bayes Classifier (with Laplace Smoothing) and run it on the training
data. Compute the training and testing accuracy.

b) Run the off-the-shelf Logistic Regression classifier. Compute the training and testing accuracy.