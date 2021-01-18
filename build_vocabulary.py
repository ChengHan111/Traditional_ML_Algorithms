 # BUild vocabulary for words seen in the email dataset

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words


vocabulary = {}
data = pd.read_csv('data/emails.csv')
nltk.download('words')
set_words = set(words.words())

def build_vocabulary(curr_email):
    idx = len(vocabulary)

    for word in curr_email:
        if word.lower() not in vocabulary and word.lower() in set_words:
            vocabulary[word] = idx
            idx += 1

if __name__ == '__main__':
    for i in range(data.shape[0]):
        curr_email = data.iloc[i,0].split()
        print(f'Current email is {i}/{data.shape[0]} and the length of vocabulary is {len(vocabulary)}')

        build_vocabulary(curr_email)
    file = open('vocabulary.txt', 'w')
    file.write(str(vocabulary))
    file.close()