import pandas as pd
import network
from collections import Counter
import numpy as np
import embeddings
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import classifier
import logging
import sys
import os
import csv


def scale_arrays(features, num_words):
    feature_size = features.shape[1]
    num_measures = int(feature_size / num_words)
    scaler = StandardScaler(with_mean=True, with_std=True)
    ini = 0
    fin = num_words
    previous = []
    for i in range(num_measures):
        sub = features[:, ini:fin]
        ini = fin
        fin += num_words
        scaled = scaler.fit_transform(sub)
        previous.append(scaled)
    result = previous[0]
    for i in range(1, num_measures):
        result = np.append(result, previous[i], axis=1)
    return result

def filter_some_books(book_csv):
    size_list = [26502, 4988, 7715, 6055, 6253, 3708, 3578, 19881, 3099]
    result = book_csv[book_csv['author'] != 'antero_quental']
    result = result.loc[~result['words_filtered'].isin(size_list)]
    return result


class AutorshipAnalysis(object):  # feature selection: commom_words , top_30_words

    #def __init__(self, iterations=10, remove_stops='con_stops', embedding_model='w2v', feature_selection='common_words', output_file='extras/'):
    def __init__(self, logger, text_partition, limiar, remove_stops='con_stops', embedding_model='w2v', feature_selection='common_words',output_file='extras/'):
        self.remove_stops = remove_stops
        #self.book_csv = pd.read_csv('books.csv')
        self.book_csv = pd.read_csv('../databases/books_authorship_english.csv')
        #self.book_csv = filter_some_books(self.book_csv) #### reducing corpusss!!!
        self.book_csv.info()
        self.embedding_model = embedding_model
        #self.embedding_percentages = [1,10,20, 30,40,50, 60]#[1,5,10,15,20,25,30]
        #self.embedding_percentages = [1,5,10,15,20,25,30]
        self.embedding_percentages = limiar
        self.feature_selection = feature_selection
        #self.iterations = iterations
        self.iterations = 0
        self.text_partition_size = text_partition ###
        self.output_file = output_file
        self.path_results = 'results/' + self.output_file + '/'
        #try:
        #    os.mkdir(self.path_results)
        #except:
        #    print("Existe")
        #log_file = self.path_results + 'log_file.log'

        #logging.basicConfig(filename=log_file, level=logging.DEBUG)
        self.logger = logger
        #self.logger = logging.getLogger('testing_portugues_autorship')
        #self.logger.setLevel(logging.DEBUG)
        #ch = logging.StreamHandler()
        #ch.setLevel(logging.DEBUG)
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #ch.setFormatter(formatter)
        #self.logger.addHandler(ch)
        #out_file = self.path_results + 'output.txt'
        #sys.stdout = open(out_file, 'w')


    def get_common_words(self, segments):
        commom_words = segments[0]
        for index, i in enumerate(segments):
            commom_words = list(set(commom_words) & set(i))
        result = {word: index for index, word in enumerate(commom_words)}
        return result

    def get_top_words(self, segments): ## 50???
        top_words = int(self.feature_selection[self.feature_selection.rfind('_')+1:])
        all_words = []
        for i in segments:
            all_words.extend(list(set(i)))

        counts = Counter(all_words)
        features = counts.most_common(top_words)
        most_commom = dict()
        #most_commom = [feat[0] for feat in features]
        for index, feat in enumerate(features):
            most_commom[feat[0]] = index
        return most_commom


    def organize_books(self):
        if self.remove_stops == 'sin_stops':
            key = 'words_filtered'
            key2 = 'filtered_content'
        else: #con_stops
            key = 'words_complete'
            key2 = 'complete_content'
        #min_size_filtered = min(list(self.book_csv[key]))
        min_size_filtered = self.text_partition_size
        print('Min book size:', min_size_filtered, '\n')
        self.logger.info('Min book size: ' + str(min_size_filtered) + '\n')

        corpus = list(self.book_csv[key2])
        corpus = [text.split() for text in corpus]
        segmented_corpus = []
        auxiliar_container = []

        size_partitions = []
        for book in corpus:
            partitions = int(round(len(book)/min_size_filtered,2)+0.5)
            segments = [book[int(round(min_size_filtered * i)): int(round(min_size_filtered * (i + 1)))] for i in range(partitions)]
            size_partitions.append(len(segments))
            segmented_corpus.append(segments)
            for i in segments:
                auxiliar_container.append(i)

        if self.feature_selection == 'common_words':
            words_features = self.get_common_words(auxiliar_container)
        else:
            words_features = self.get_top_words(auxiliar_container)
        self.iterations = int(np.mean(size_partitions))
        return corpus, segmented_corpus, words_features


    def analysis(self):
        corpus, corpus_partitions, words_features = self.organize_books()
        #corpus_partitions = corpus_partitions[0:3]
        authors = list(self.book_csv['author'])
        total_authors = list(set(self.book_csv['author']))
        number_books = (self.book_csv[self.book_csv['author'] == total_authors[0]]).shape[0]

        print('Word features:', words_features)
        self.logger.info(' Word features: ' + str(words_features) )

        self.logger.info('Training word embeddings ....')
        objEmb = embeddings.WordEmbeddings(corpus, self.embedding_model)
        model = objEmb.get_embedding_model()
        self.logger.info('Word embeddings sucessfully trained')

        dict_authors = list(set(authors))
        dict_authors = {author:index for index,author in enumerate(dict_authors)}

        iteration_scores = []
        for iteration in range(1):
        #for iteration in range(self.iterations):#
            print('Init of iteration:', iteration+1)
            self.logger.info('Init of iteration: ' + str(iteration+1))

            all_network_features = []
            labels = []

            for index, (book, author) in enumerate(zip(corpus_partitions, authors)):
                random_index = random.randint(0, len(book) - 1)
                print('book:', index + 1)
                print('author:', author)
                print('partitions:', len(book))
                print('iteration: ' + str(iteration+1) + ' of ' + str(self.iterations))

                labels.append(dict_authors[author])
                #selected_partition = book[random_index]
                selected_partition = book[0]


                obj = network.CNetwork(selected_partition, model, self.embedding_percentages, self.path_results)
                cNetworks = obj.create_networks()  # network_normal , 1%, 2%, 3%, 4%, ... 20%
                #cNetworks = obj.create_filtered_networks()
                network_features = [obj.get_network_measures(net, words_features) for net in cNetworks]
                all_network_features.append(network_features)
                print('\n')

            all_network_features = np.array(all_network_features)
            print (all_network_features.shape)
            self.logger.info('shape:'+ str(all_network_features.shape))
            scaler = StandardScaler(with_mean=True, with_std=True)
            all_scores = []

            for index in range(len(self.embedding_percentages)+1):
                limiar_features = all_network_features[:, index]
                limiar_features = scaler.fit_transform(limiar_features)
                #limiar_features = scale_arrays(limiar_features, len(words_features))
                obj = classifier.Classification(limiar_features, labels, number_books)
                scores = obj.get_scores()
                all_scores.append(scores)
                print(index,limiar_features.shape, labels)
                print()

            iteration_scores.append(all_scores)
            print('------ End of iteration ' + str(iteration+1) + ' ------\n\n')
            self.logger.info('------ End of iteration ' + str(iteration+1) + ' ------\n\n')

        print('Final results .....')
        self.logger.info('Final results .....')
        iteration_scores = np.array(iteration_scores)

        percs = [str(val) + '%' for val in self.embedding_percentages]
        percs.insert(0, '0%')
        #self.logger.info('Results for texts of ' + str(self.text_partition_size) + ' words')
        print('Results for texts of ' + str(self.text_partition_size) + ' words')
        final_results = []
        for limiar in range(len(self.embedding_percentages)+1):
            values = iteration_scores[:,limiar]
            scores = values.mean(axis=0)
            scores = [round(score, 2) for score in scores]
            final_results.append(scores)
            print('Final scores:', percs[limiar], scores)
            #self.logger.info('Final scores: ' + str(percs[limiar]) + ' --> ' + str(scores))
        return final_results


    def analysis2(self):
        corpus, corpus_partitions, words_features = self.organize_books()
        authors = list(self.book_csv['author'])
        total_authors = list(set(self.book_csv['author']))
        number_books = (self.book_csv[self.book_csv['author'] == total_authors[0]]).shape[0]
        print('Word features:', words_features)

        dict_authors = list(set(authors))
        dict_authors = {author: index for index, author in enumerate(dict_authors)}

        all_network_features = []
        labels = []


        for index, (book, author) in enumerate(zip(corpus_partitions, authors)):
            random_index = random.randint(0, len(book) - 1)
            print('book:', index + 1)
            print('author:', author)
            print('partitions:', len(book))
            labels.append(dict_authors[author])
            selected_partition = book[0]

            obj = network.CNetwork(selected_partition, model, self.embedding_percentages, self.path_results)
            cNetwork = obj.create_network()
            network_features = obj.get_network_measures(cNetwork, words_features)
            all_network_features.append(network_features)

        all_network_features = np.array(all_network_features)
        print(all_network_features.shape)
        scaler = StandardScaler(with_mean=True, with_std=True)
        all_network_features =  scaler.fit_transform(all_network_features)


        with open('test_file.csv', mode='w') as myFile:
            writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['vector', 'label', 'author'])
            for feature, label, author in zip(all_network_features, labels, authors):
                vec = feature.tolist()
                print(label)
                writer.writerow([vec, label, author])
                print()


if __name__ == '__main__':

    # obj = AutorshipAnalysis(iterations=iterations, remove_stops=use_stops, embedding_model=model, feature_selection=word_selection, output_file=folder_results)
    # obj.analysis()


    #arguments = sys.argv
    model = 'ft'  # # ft
    use_stops = 'sin_stops'  # 2
    word_selection = 'top_100'  # 3
    folder_results = model  # 4

    # text_sizes = [1000, 1500, 2000, 2500, 5000]
    #text_sizes = [1000, 1500, 2000, 2500, 5000, 10000]
    text_sizes = [1000, 1500]
    #limiars = [1, 5, 10, 15, 20, 25, 30, 35,40,45,50]
    limiars = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


    path_results = 'results/' + folder_results + '/'
    try:
        os.mkdir(path_results)
    except:
        print("Existe")

    logger = logging.getLogger('testing_portugues_autorship')

    obj = AutorshipAnalysis(logger=logger, text_partition=1000, limiar=limiars, remove_stops=use_stops, embedding_model=model, feature_selection=word_selection, output_file=folder_results)
    obj.analysis2() # 604 pid






