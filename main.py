import os
from nltk.tokenize import word_tokenize
import string
import re
import nltk
#from morphological_analysis import lemma
import csv
import pandas as pd
from gensim.models import KeyedVectors


'''
Processing books and saving into csv file
'''


'''
def read_book(path):
    arc = open(path, 'r', encoding='utf-8-sig')
    book = arc.read()
    book = book.lower()
    book = " ".join(book.split())
    for c in string.punctuation:
        book =  book.replace(c, "")
    book = ''.join([i for i in book if not i.isdigit()])

    words = word_tokenize(book, language='portuguese')
    words = [lemma(word) for word in words]
    filtered_words = [word for word in words if word not in stopwords and len(word)>2]

    str_words = ' '.join(words)
    str_filtered = ' '.join(filtered_words)

    return str_words, len(words),str_filtered, len(filtered_words)




database = '../databases/portuguese/'
authors = os.listdir(database)
authors.remove('.DS_Store')
authors.remove('jos_almada')


stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords = {i : x for x, i in enumerate(stopwords)}


headers = ['author', 'book_id', 'complete_content', 'words_complete', 'filtered_content', 'words_filtered']
with open('books.csv', mode='w') as myFile:
    writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)

    for count, author in enumerate(authors):
        books = os.listdir(database+author)
        print(count+1,author)
        print(books)
        for book in books:
            content, len_content, content_filtered, len_filtered = read_book(database+author+'/'+book)
            book_id = book[:book.rfind('.txt')]
            print(author, book_id, len_content, len_filtered)
            row = [author, book_id, content, len_content, content_filtered, len_filtered]
            writer.writerow(row)
    print()



'''
'''
import numpy as np
from sklearn.preprocessing import StandardScaler

features = [[1,2,3,4,5,6,7,8,0,1,2,3],
            [11,22,33,41,5,6,1,4,4,3,2,1],
            [10,1,22,3,14,5,11,9,1,1,1,1],
            [15,4,23,2,11,10,12,1,2,3,1,23],
            [91,18,71,6,5,4,4,8,0,0,1,2],
            ]


features = np.array(features)
num_words = 4
feature_size = features.shape[1]
num_measures = int(feature_size/num_words)
num_documents = features.shape[0]
print('feature size:',feature_size)
print('num measures:',num_measures)
print('num words:', num_words)
print('num documents:', num_documents)
print()

print(features)

print('\n')

print('Testing ...')


scaler = StandardScaler(with_mean=True, with_std=True)
ini = 0
fin = num_words
previous = []
for i in range(num_measures):
    sub = features[:,ini:fin]
    ini = fin
    fin+=num_words
    scaled = scaler.fit_transform(sub)
    previous.append(scaled)
    print()

first = previous[0]
for i in range(1,num_measures):
    first = np.append(first, previous[i], axis=1)

print('haberrr final::')
print(first)
'''

'''
book_csv = pd.read_csv('../databases/books_authorship_english.csv')
corpus = list(book_csv['complete_content'])

output_file = open('extras/auxiliar_english.txt', 'w')
for i in corpus:
    output_file.write(i + '\n')
'''

'''
import fasttext
path = 'extras/auxiliar.txt'
model = fasttext.train_unsupervised(path, model='skipgram')#, dim=100)

print(model.words)   # list of words in dictionary
print(len(model['sympathia']),model['sympathia'])
'''

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import umap



haber = pd.read_csv('test_file.csv')

haber.info()

vectores = list(haber['vector'])
labels = list(haber['label'])

result = []
for index, vec in enumerate(vectores):
    vec = vec.replace('[', '')
    vec = vec.replace(']', '')
    vec = vec.split(',')
    vec = [float(val) for val in vec]
    result.append(vec)
    #print(index, len(vec))

result = np.array(result)
labels = np.array(labels)
authores = np.array(haber['author'])

dict_auts = {label:aut for label, aut in zip(labels, authores)}


reducer = umap.UMAP()
embedding = reducer.fit_transform(result)
print (embedding.shape)
#print(embedding)




colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
            'pink', 'gray', 'olive', 'cyan', 'black', 'navy',
            'violet']


legends = {'hector_hugh':'HH', 'thomas_hardy':'TH', 'daniel_defoe':'DD', 'alan_poe':'AP', 'bram_stoker':'BS',
           'mark_twain':'MT', 'charles_dickens':'CDi', 'pelham_grenville':'PG', 'charles_darwin':'CDa',
           'arthur_doyle':'AD', 'george_eliot':'GE', 'jane_austen':'JA', 'joseph_conrad':'JC'}



scatter_x = embedding[:, 0]
scatter_y = embedding[:, 1]
#group = authores
group = labels
cdict = {count:col for count, col in enumerate(colors)}
fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = legends[dict_auts[g]], s = 100)
plt.gca().set_aspect('equal', 'datalim')
ax.legend(loc='upper right')
plt.title('Author distribution', fontsize=24)
plt.show()

#plt.scatter(embedding[:, 0], embedding[:, 1], c=[colors[x] for x in labels])
#plt.gca().set_aspect('equal', 'datalim')
#plt.title('Author books', fontsize=24)
#plt.legend()
#plt.show()



