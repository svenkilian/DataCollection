# Standalone executable module implementing experimental NLP techniques to textual information.
import re
import time

from sklearn.svm import LinearSVC, SVC

from config import ROOT_DIR
import json
import os
import pickle
import sys
from pprint import pprint

import gensim
import pandas as pd
import pycountry as country
import pymongo
import spacy
from bson.json_util import dumps
from gensim import corpora
from polyglot.text import Text
from tabulate import tabulate
from tqdm import tqdm
from urllib.parse import urlparse
import DataCollection
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout
from keras.optimizers import Adam

from NLPFunctions import perform_lda, text_preprocessing, prepare_text_for_lda, LemmaTokenizer


def get_stats(data_frame):
    """
    Prints basic statistics of repositories in data frame to console.

    :param data_frame: Data frame object to analyse
    :return:
    """

    # JOB: Add aggregate columns
    data_frame['repo_created_at'] = pd.to_datetime(data_frame['repo_created_at'], infer_datetime_format=True)
    data_frame['repo_last_mod'] = pd.to_datetime(data_frame['repo_last_mod'], infer_datetime_format=True)

    data_frame['extracted_architecture'] = data_frame['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))
    data_frame['model_file_found'] = data_frame['py_data'].apply(func=lambda x: x.get('model_file_found'))
    data_frame['imports_keras'] = data_frame['py_data'].apply(func=lambda x: x.get('imports_keras'))
    data_frame['has_architecture_info'] = data_frame.apply(
        func=lambda x: x.get('extracted_architecture') or x.get('model_file_found'), axis=1)
    data_frame['has_topics'] = data_frame['repo_tags'].apply(func=lambda x: len(x) is not 0 if x is not None else False)

    # Define year and month for grouping
    per_year = data_frame.repo_created_at.dt.to_period('Y')
    per_month = data_frame.repo_created_at.dt.to_period('M')

    # Per year view
    repo_count_per_year = data_frame.groupby(per_year).size().reset_index(
        name='Counts of repos').set_index('repo_created_at')
    print(repo_count_per_year)
    print(2 * '\n')

    # Per month view
    repo_count_per_month = data_frame.groupby(per_month).size().reset_index(
        name='Counts of repos').set_index('repo_created_at')
    print(repo_count_per_month)
    print(2 * '\n')

    # Keras used view
    keras_used_count = data_frame.groupby(['keras_used']).size().reset_index(name='Counts of repos').set_index(
        'keras_used')
    print(keras_used_count)
    print(2 * '\n')

    # Has H5 view
    has_h5_count = data_frame.groupby(['has_h5']).size().reset_index(name='Counts of repos').set_index(
        'has_h5')
    print(has_h5_count)
    print(2 * '\n')

    # Has extracted architecture view
    extracted_architecture_count = data_frame.groupby(['extracted_architecture']).size().reset_index(name=
                                                                                                     'Counts of repos').set_index(
        'extracted_architecture')
    print(extracted_architecture_count)
    print(2 * '\n')

    # Model file found view
    model_file_found_count = data_frame.groupby(['model_file_found']).size().reset_index(
        name='Counts of repos').set_index('model_file_found')
    print(model_file_found_count)
    print(2 * '\n')

    # Imports Keras view
    py_data_imports_keras_count = data_frame.groupby(['imports_keras']).size().reset_index(
        name='Counts of repos').set_index('imports_keras')
    print(py_data_imports_keras_count)
    print(2 * '\n')

    # Has architecture info view
    has_architecture_info = data_frame.groupby(['has_architecture_info']).size().reset_index(
        name='Counts of repos').set_index('has_architecture_info')
    print(has_architecture_info)
    print(2 * '\n')

    # Topic view
    has_topic_info = data_frame.groupby(['has_topics']).size().reset_index(
        name='Counts of repos').set_index('has_topics')
    print(has_topic_info)
    print(2 * '\n')

    # print(tabulate(data_frame[data_frame['reference_list'].astype(str) != '[]'].sample(20), headers='keys',
    #                tablefmt='psql', showindex=True))


def analyze_references(data_frame):
    """
    Shows frequency statistics of http link occurrences.

    :param data_frame: Data frame containing links as Series titled "reference_list" and "see_also_links"
    :return:
    """

    # Aggregate columns to form a new column with all links
    data_frame['all_links'] = data_frame.apply(
        lambda row: np.concatenate((np.asarray(row['reference_list']), np.asarray(row['see_also_links']))), axis=1)

    # Print sample of new data frame
    print(tabulate(data_frame.sample(20)['all_links'], tablefmt='psql', showindex=True))

    # Get list of all links from new data frame column
    link_list = np.concatenate(([links for links in data_frame['all_links']]))
    link_list = [urlparse(link).netloc for link in link_list]

    # Generate Counter object and cast to dict
    counter = dict(Counter(link_list))

    # Create new data frame containing count information
    df = pd.DataFrame(counter, index=['Count']).transpose().sort_values(['Count'], ascending=False)

    # Print new data frame
    print(df)


def analyze_topics(data_frame):
    """
    Shows frequency statistics of topics and returns array of neural network types

    :param data_frame: Data frame containing topics
    :return: Array with neural network types
    """

    # Replace 'None' values with empty array
    # data_frame['repo_tags'].replace(pd.np.nan, np.empty_like(), inplace=True)

    topic_list = []

    # Get list of all links from new data frame column
    for row in data_frame['repo_tags']:
        if row is not None:
            topic_list.extend(row)

    # Generate Counter object and cast to dict
    counter = dict(Counter(topic_list))

    # Create new data frame containing count information
    df = pd.DataFrame.from_dict(counter, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['Tag', 'Count']
    df.sort_values(['Count'], ascending=False, inplace=True)

    # Clean out redundant topics
    stop_topics = {'keras', 'deep-learning', 'tensorflow', 'machine-learning', 'python', 'neural-network',
                   'keras-tensorflow', 'python3', 'deep-neural-networks', 'neural-networks', 'pytorch',
                   'keras-neural-networks', 'flask', 'mnist', 'numpy', 'scikit-learn', 'keras-models', 'kaggle'}
    type_list = {'convolutional-neural-networks', 'cnn', 'lstm', 'rnn', 'cnn-keras', 'recurrent-neural-networks',
                 'lstm-neural-networks', 'generative-adversarial-network', 'gan', 'reinforcement-learning',
                 'deep-reinforcement-learning', 'autoencoder'}

    # Filter by repositories with topic label from type_list
    # df = df[df['Tag'].isin(type_list)]
    df = df[df['Tag'].isin(type_list)]

    # Print new data frame
    # print(df.head(60))

    return df['Tag'].tolist()


def get_readme_langs(df):
    """
    Identifies langage (name and code) for all readme texts in given data frame

    :param df: Data frame to extract languages and language codes from
    :return: New data frame with two added columns for language name and code
    """
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            if row['readme_text'] is not ('' or None):
                text = Text(str(row['readme_text']))
                language_code = text.language.code
                if language_code is not None:
                    language_name = country.languages.get(alpha_2=language_code).name
                else:
                    language_name = None
            else:
                language_name = None
                language_code = None
        except AttributeError as ae:
            language_name = None
            language_code = None

        # Add extracted language information to data frame
        df.at[index, 'language_readme'] = language_name
        df.at[index, 'language_readme_code'] = language_code

    return df


def type_encoding(df, type_list):
    """
    Performs multi-label binary encoding for neural network type.
    :param df: Data frame to use
    :param type_list: List of neural network types
    :return:
    """

    type_match_dict = {'convolutional-neural-networks': 'cnn',
                       'cnn-keras': 'cnn',
                       'cnns': 'cnn',
                       re.compile('^.*cnn.*$'): 'cnn',
                       re.compile('^.*convolution.*$'): 'cnn',
                       'recurrent-neural-networks': 'rnn',
                       'generative-adversarial-network': 'gan',
                       'deep-reinforcement-learning': 'reinforcement-learning',
                       'lstm-neural-networks': 'lstm'
                       }

    # Deduplicate type_list
    type_list = [item for item in type_list if item not in type_match_dict.keys()]

    # Replace synonymous types
    df['repo_tags'] = data_frame['repo_tags'].apply(
        lambda topic_list: [type_match_dict.get(topic, topic) for topic in
                            topic_list] if topic_list is not None else [])

    for nn_type in type_list:
        df[nn_type] = pd.Series(
            [1 if nn_type in (topic_list if topic_list is not None else []) else 0 for topic_list in
             df['repo_tags'].tolist()])

    df = df[df[type_list].any(axis=1)]

    return df, type_list


def get_data_from_collection(path_to_data, collection_name):
    """
    Retrieve data from database collection and store locally to file.
    :param path_to_data: Path to export location
    :param collection_name: Name of location to retrieve
    :return:
    """
    # Create collection object
    collection = DataCollection.DataCollection(collection_name).collection_object

    print('Downloading data from database ...')
    # JOB: Save database query result to json
    data = dumps(collection.find({}))

    print('Write data to file ...')
    with open(path_to_data, 'w') as file:
        file.write(data)


def get_train_test_data(df):
    """
    Applies pre-processing and returns train-test split of data set.
    :param load_data: Flag indicating whether data is loaded from file
    :param df: Data frame to process
    :return: Train-test split of preprocessed data
    """

    X = df['readme_text']
    y = df.iloc[:, 1:]

    # Fit count vectorizer and transform features
    begin_time = time.time()
    X = CountVectorizer(strip_accents='unicode', ngram_range=(1, 2),
                        tokenizer=LemmaTokenizer()).fit_transform(X)
    # X = TfidfTransformer(use_idf=True).fit_transform(X)
    end_time = time.time()
    print('Preprocessing/Vectorizing duration: %g sesconds' % round((end_time - begin_time), 2))

    # JOB: Apply Latent Dirichlet Allocation

    # lda = LatentDirichletAllocation(n_components=6, random_state=0, learning_method='online', learning_decay=0.6,
    #                                 max_doc_update_iter=200, verbose=1)

    print('Shape of data after vectorizing:')
    print(X.shape)
    print(y.shape)

    # Obtain train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains and returns model.
    :param X_train: Features of training set
    :param y_train: Labels of training set
    :return: Trained classifier
    """
    # classifier = BinaryRelevance(GaussianNB())
    # classifier = BinaryRelevance(LinearSVC())

    # Instantiate multilabel classifier
    # classifier = BinaryRelevance(LinearSVC())

    # clf = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=12)
    # Grid search result: C = 1, tol = 1

    # classifier.fit(X_train.astype(float), y_train.astype(float))

    # model = Sequential()
    # model.add(Embedding(max_words, 20, input_length=maxlen))
    # model.add(Dropout(0.15))
    # model.add(GlobalMaxPool1D())
    # model.add(Dense(num_classes, activation='sigmoid'))
    #
    # model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    # callbacks = [
    #     ReduceLROnPlateau(),
    #     EarlyStopping(patience=4),
    #     ModelCheckpoint(filepath='model-simple.h5', save_best_only=True)
    # ]
    #
    # history = model.fit(x_train, y_train,
    #                     class_weight=class_weight,
    #                     epochs=20,
    #                     batch_size=32,
    #                     validation_split=0.1
    # callbacks = callbacks)
    # return None


def test_model(df):
    """
    Tests model performance on test data.
    :param clf: Trained classifier
    :param X: Test data features
    :param y: Test data labels
    :return: Accuray score
    """
    # predictions = clf.predict(X.astype(float)).toarray()
    X = df['readme_text']
    y = df.iloc[:, 1:]

    # Fit count vectorizer and transform features
    X = CountVectorizer(strip_accents='unicode', ngram_range=(1, 2),
                        tokenizer=LemmaTokenizer()).fit_transform(X)

    classifier = BinaryRelevance(LinearSVC(max_iter=10000))
    mean_accuracy = cross_val_score(classifier, X.astype(float), y.astype(float), cv=5, n_jobs=12, verbose=3)

    return mean_accuracy


if __name__ == '__main__':
    """
    Main method
    """

    # Specify path to saved repository data
    path_to_data = os.path.join(ROOT_DIR, 'DataCollection/data/data.json')

    # Download data from database (only perform when data changed)
    # get_data_from_collection(path_to_data, 'Repos_Exp')

    # Load json data as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # Make DataFrame from json
    data_frame = pd.DataFrame(data)

    # Extract list of Neural Network types from labels
    type_list = analyze_topics(data_frame)

    # Encode neural network types with multti-level binarization and deudplicate type_list
    df, type_list = type_encoding(data_frame, type_list)

    # print(tabulate(df[['readme_text', 'repo_tags', *type_list]].sample(20), headers='keys',
    #                tablefmt='psql', showindex=True))

    # print(tabulate(
    #     df[['repo_tags', *type_list, 'n_classes']].sort_values(['n_classes'], ascending=False).head(10),
    #     headers='keys',
    #     tablefmt='psql', showindex=True))

    # print(df[type_list].sum())

    # Filter by non-empty English-language readme texts
    df_learn = df[(df['readme_language'] == 'English') & (df['readme_text'] != '') & (df['readme_text'] != None)][
        ['readme_text', *type_list]]

    X_train, X_test, y_train, y_test = get_train_test_data(df_learn)

    # print(tabulate(X_train.sample(20), headers='keys',
    #                tablefmt='psql', showindex=True))
    #

    print('Number of repositories: %d' % df_learn.shape[0])

    print('Training model ...')
    # begin_time = time.time()
    # clf = train_model(X_train, y_train)
    # end_time = time.time()
    # print('Model fit duration: %g sesconds' % round((end_time - begin_time), 2))

    test_score = test_model(df_learn)

    print('Test accuracy score: %s' % test_score)

    # prediction = clf.predict(X_test.astype(float)).toarray()

    # topics = perform_lda(data_frame)

    # Print words associated with latent topics to console
    # print('\n' * 5)
    # for i, topic in enumerate(topics):
    #     print('Topic %d: %s' % (i, topic))

    # JOB: Stop execution of method here
    sys.exit()  # Comment
