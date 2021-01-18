# Go through each email and map into X,y dataset

# we are trying to map each email into some frequency array
# and this is gonna be the input of naive bayes
import pandas as pd
import numpy as np
import ast

data = pd.read_csv('data/emails.csv')
file = open('vocabulary.txt', 'r')
contents = file.read()
vocabulary = ast.literal_eval(contents)


X = np.zeros((data.shape[0], len(vocabulary)))
y = np.zeros((data.shape[0]))

for i in range(data.shape[0]):
    email = data.iloc[i, 0].split()

    for email_word in email:
        if email_word.lower() in vocabulary:
            X[i, vocabulary[email_word]] += 1
            y[i] = data.iloc[i, 1]
# print(X[0,:])
# print(data.iloc[2500, 0])
np.save('data/X.npy', X)
np.save('data/y.npy', y)