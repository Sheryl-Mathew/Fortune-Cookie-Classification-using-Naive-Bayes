import pandas as pd
import numpy as np 
from functools import reduce
import operator
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

f = open('output.txt', 'a+')

def read_files():
    train_data = pd.read_csv("traindata.txt",header=None)
    train_labels = pd.read_csv("trainlabels.txt",header=None)
    test_data = pd.read_csv("testdata.txt",header=None)
    test_labels = pd.read_csv("testlabels.txt",header=None)
    stopwords_list = pd.read_csv("stoplist.txt",header=None)
    return train_data,train_labels,test_data,test_labels,stopwords_list

def convert_to_list(dataset):
    data = dataset.values.tolist()
    data = reduce(operator.add, data)
    words = []
    for value in data:
        word = value.split()
        words.append(word)
    return words

def remove_stopwords(dataset,stopwords):
    all_data = []
    data = []
    dataset = reduce(operator.add, dataset)
    stopwords = reduce(operator.add, stopwords)
    for word in dataset: 
        if word not in stopwords:
            all_data.append(word)
    for word in all_data:
        if word not in data:
            data.append(word)
    data.sort()
    return data

def feature_matrix(dataset,vocabulary):
    data = []
    for row_data in dataset:
        word_list = dict.fromkeys(vocabulary, 0)
        for word in row_data:
            if word in word_list:
                word_list[word] = 1
        data.append(word_list)
    feature_matrix = pd.DataFrame(data)
    return feature_matrix

def null_check(value):
    if value == 0:
        return 0
    else:
        return value

def calculate_probability_laplace_smoothing(value,total_count):
    probability = null_check((value+1)/(total_count+2))
    return probability

def calculate_probability(value,total_count):
    probability = null_check(value/total_count)
    return probability

def calculate_wise_future(dataframe,target,column_name,value):
    dataframe['target'] = target
    if value is None:
        wise = len(dataframe.loc[dataframe['target'] == 0])
        future = len(dataframe.loc[dataframe['target'] == 1])
    else:
        wise = len(dataframe.loc[(dataframe[column_name] == value) & (dataframe['target'] == 0)])
        future = len(dataframe.loc[(dataframe[column_name] == value) & (dataframe['target'] == 1)])
    return wise,future,wise+future
   
def probability_words(feature_matrix,vocabulary,target):
    probability_dictionary = {}
    for word in vocabulary:
        dictionary = {word:{'present':{'wise':{},'future':{}},'absent':{'wise':{},'future':{}}}}
        for value in [0,1]:
            number_of_wise,number_of_future,total = calculate_wise_future(feature_matrix,target,word,value)
            probability_wise = calculate_probability_laplace_smoothing(number_of_wise,total)
            probability_future = calculate_probability_laplace_smoothing(number_of_future,total)
            if value == 1:
                dictionary[word]['present']['wise'] = probability_wise
                dictionary[word]['present']['future'] = probability_future
            else:
                dictionary[word]['absent']['wise'] = probability_wise
                dictionary[word]['absent']['future'] = probability_future
        probability_dictionary.update(dictionary)

    number_of_wise,number_of_future,total = calculate_wise_future(feature_matrix,target,None,None)
    probability_wise_total = calculate_probability(number_of_wise,total)
    probability_future_total = calculate_probability(number_of_future,total)

    return probability_dictionary,probability_wise_total,probability_future_total

def naive_bayes(probability_dictionary,feature_matrix,probability_wise_total,probability_future_total,type_of_data):
    class_labels = feature_matrix.iloc[:,-1]
    feature_matrix = feature_matrix.drop('target',1)
    number_of_rows = feature_matrix.shape[0]
    words = probability_dictionary.keys()
    number_of_correct_predictions = 0
    number_of_incorrect_predictions = 0
    for index in range(number_of_rows):
        probability_wise = 1
        probability_future = 1
        data_to_traverse = feature_matrix.iloc[index]
        for word in words:
            if data_to_traverse[word] == 1:
                probability_wise*=probability_dictionary[word]['present']['wise']
                probability_future*=probability_dictionary[word]['present']['future']

        wise_prob = probability_wise*probability_wise_total
        future_prob = probability_future*probability_future_total
        if wise_prob > future_prob:
            label = 0
        else:
            label = 1
        if(label == class_labels.iloc[index]):
            number_of_correct_predictions+=1
        else:
            number_of_incorrect_predictions+=1
    accuracy = calculate_probability(number_of_correct_predictions,number_of_rows)
    print("Accuracy of %s data using Naive Bayes from Scratch: %f" %(type_of_data, accuracy*100),file = f)

def naive_bayes_sklearn(training_data,testing_data,type_of_data):
    training_class_label = training_data.iloc[:,-1]
    testing_class_label = testing_data.iloc[:,-1]
    training_data = training_data.drop('target',1)
    testing_data = testing_data.drop('target',1)
    model = MultinomialNB().fit(training_data, training_class_label)
    predicted = model.predict(testing_data)
    accuracy = accuracy_score(testing_class_label,predicted)
    print("Accuracy of %s data using Naive Bayes from Sklearn Multinomial: %f" %(type_of_data, accuracy*100),file = f)

def logistic_regression_sklearn(training_data,testing_data,type_of_data):
    training_class_label = training_data.iloc[:,-1]
    testing_class_label = testing_data.iloc[:,-1]
    training_data = training_data.drop('target',1)
    testing_data = testing_data.drop('target',1)
    model = LogisticRegression().fit(training_data, training_class_label)
    predicted = model.predict(testing_data)
    accuracy = accuracy_score(testing_class_label,predicted)
    print("Accuracy of %s data using Logistic Regression from Sklearn: %f" %(type_of_data, accuracy*100), file =f)

train_data,train_label,test_data,test_label,stopwords_list = read_files()
train_dictionary = convert_to_list(train_data)
test_dictionary = convert_to_list(test_data)
stopwords = convert_to_list(stopwords_list)
vocabulary = remove_stopwords(train_dictionary,stopwords)
feature_matrix_train = feature_matrix(train_dictionary,vocabulary)
feature_matrix_test = feature_matrix(test_dictionary,vocabulary)
feature_matrix_test['target'] = test_label
probability_dictionary,probability_wise_total,probability_future_total = probability_words(feature_matrix_train,vocabulary,train_label)


print("Naive Bayes:",file =f)
print(file =f)
naive_bayes(probability_dictionary,feature_matrix_train,probability_wise_total,probability_future_total,"training")
naive_bayes(probability_dictionary,feature_matrix_test,probability_wise_total,probability_future_total,"testing")
print(file =f)
naive_bayes_sklearn(feature_matrix_train,feature_matrix_train,"training")
naive_bayes_sklearn(feature_matrix_train,feature_matrix_test,"testing")
print(file =f)
print("Logistic Regression:",file =f)
print(file =f)
logistic_regression_sklearn(feature_matrix_train,feature_matrix_train,"training")
logistic_regression_sklearn(feature_matrix_train,feature_matrix_test,"testing")

f.close()