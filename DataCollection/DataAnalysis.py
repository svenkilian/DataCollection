# Standalone executable module implementing experimental NLP techniques to textual information.

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
from config import ROOT_DIR


def get_stats(data_frame):
    """
    Prints basic statistics of repositories in data frame to console.

    :param data_frame: Data frame object to analyse
    :return:
    """

    per_year = data_frame.repo_created_at.dt.to_period('Y')
    per_month = data_frame.repo_created_at.dt.to_period('M')

    repo_count_per_year = data_frame.groupby(per_year).size().reset_index(
        name='Counts of repos').set_index('repo_created_at')
    print(repo_count_per_year)
    print(2 * '\n')

    repo_count_per_month = data_frame.groupby(per_month).size().reset_index(
        name='Counts of repos').set_index('repo_created_at')
    print(repo_count_per_month)
    print(2 * '\n')

    keras_used_count = data_frame.groupby(['keras_used']).size().reset_index(name='Counts of repos').set_index(
        'keras_used')
    print(keras_used_count)
    print(2 * '\n')

    has_h5_count = data_frame.groupby(['has_h5']).size().reset_index(name='Counts of repos').set_index(
        'has_h5')
    print(has_h5_count)
    print(2 * '\n')

    extracted_architecture_count = data_frame.groupby(['extracted_architecture']).size().reset_index(name=
        'Counts of repos').set_index('extracted_architecture')
    print(extracted_architecture_count)
    print(2 * '\n')

    model_file_found_count = data_frame.groupby(['model_file_found']).size().reset_index(
        name='Counts of repos').set_index('model_file_found')
    print(model_file_found_count)
    print(2 * '\n')

    py_data_imports_keras_count = data_frame.groupby(['imports_keras']).size().reset_index(
        name='Counts of repos').set_index('imports_keras')
    print(py_data_imports_keras_count)
    print(2 * '\n')

    has_architecture_info = data_frame.groupby(['has_architecture_info']).size().reset_index(
        name='Counts of repos').set_index('has_architecture_info')
    print(has_architecture_info)
    print(2 * '\n')

    has_wiki_count = data_frame.groupby(['has_wiki']).size().reset_index(name='Counts of repos').set_index(
        'has_wiki')
    print(has_wiki_count)
    print(2 * '\n')

    # owner_count = data_frame.groupby(['repo_owner']).size().reset_index(name='Counts of repos').sort_values(
    #     by=['Counts of repos'], ascending=False)
    # print(owner_count)

    # print(owner_count[owner_count['Counts of repos'] > 1].sum())

    print(tabulate(data_frame[data_frame['reference_list'].astype(str) != '[]'].sample(20), headers='keys', tablefmt='psql', showindex=True))


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
    print(tabulate(data_frame.sample(20), headers='keys', tablefmt='psql', showindex=True))

    # Get list of all links from new data frame column
    link_list = np.concatenate(([links for links in data_frame['all_links']]))
    link_list = [urlparse(link).netloc for link in link_list]

    # Generate Counter object and cast to dict
    counter = dict(Counter(link_list))

    # Create new data frame containing count information
    df = pd.DataFrame(counter, index=['Count']).transpose().sort_values(['Count'], ascending=False)

    # Print new data frame
    print(df)


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


if __name__ == '__main__':
    """
    Main method to be executed when module is run.
    """

    # Specify path to saved repository data
    path_to_data = os.path.join(ROOT_DIR, 'DataCollection/data/data.json')

    # Create collection object
    collection = DataCollection.DataCollection('Repos_Exp').collection_object

    # JOB: Save database query result to json
    data = dumps(collection.find({}))

    with open(path_to_data, 'w') as file:
        file.write(data)

    # JOB: Load json data as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # JOB: Make DataFrame from json
    data_frame = pd.DataFrame(data)

    # JOB: Convert data columns
    data_frame['repo_created_at'] = pd.to_datetime(data_frame['repo_created_at'], infer_datetime_format=True)
    data_frame['repo_last_mod'] = pd.to_datetime(data_frame['repo_last_mod'], infer_datetime_format=True)

    data_frame['extracted_architecture'] = data_frame['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))
    data_frame['model_file_found'] = data_frame['py_data'].apply(func=lambda x: x.get('model_file_found'))
    data_frame['imports_keras'] = data_frame['py_data'].apply(func=lambda x: x.get('imports_keras'))
    data_frame['has_architecture_info'] = data_frame.apply(
        func=lambda x: x.get('extracted_architecture') or x.get('model_file_found'), axis=1)

    # JOB: Call method
    # analyze_references(data_frame)

    print('Number of repositories: %d' % data_frame.shape[0])

    get_stats(data_frame)
    # JOB: Stop execution of method here
    sys.exit()  # Comment

    # Print out part of data frame to console
    print(data_frame[['repo_full_name', 'readme_text']].iloc[20:30])

    # Specify path to GitHub access tokens
    access_path = os.path.join(ROOT_DIR, 'DataCollection/credentials/GitHub_Access_Token.txt')

    # JOB: Get language for readme strings
    data_frame = get_readme_langs(data_frame)

    # Print readme text and extracted language information to console
    print(data_frame[data_frame['language_readme'] != 'English'].loc[:, ['readme_text', 'language_readme']])

    # Specify readme language as categorical feature
    data_frame['language_readme'].astype('category')

    # Print language distribution (top 10 occurrences) to console
    print('\n\n')
    print(data_frame.groupby(['language_readme']).size().reset_index(name='Count').sort_values(
        'Count')[-10:])

    # Filter by English readmes
    data_en = data_frame[data_frame['language_readme'] == 'English'][:]

    # # Export data frame to excel file
    # data_en.to_excel(r'Output.xlsx', engine='xlsxwriter')

    # JOB: Apply preprocssing for topic modeling
    spacy.load('en')  # Load English language corpus
    # nltk.download('wordnet')  # Download wordnet
    # nltk.download('stopwords')  # Download stopwords

    # # Initialize empty array of text data
    # text_data = []
    #
    # # Fill text data array
    # for index, row in tqdm(data_en.iterrows(), total=data_en.shape[0]):
    #     tokens = prepare_text_for_lda(row['readme_text'])
    #     if random.random() >= 0:
    #         # print(tokens)
    #         text_data.append(tokens)
    #
    # # Create dictionary from data
    # dictionary = corpora.Dictionary(text_data)
    # # Convert dictionary into bag of words corpus
    # corpus = [dictionary.doc2bow(text) for text in text_data]
    #
    # # Save corpus and dictionary
    # pickle.dump(corpus, open('./data/corpus.pkl', 'wb'))
    # dictionary.save('./data/dictionary.gensim')

    # Load corpus and dictionary
    corpus = pickle.load(open('./data/corpus.pkl', 'rb'))
    dictionary = corpora.Dictionary.load('./data/dictionary.gensim')

    # Extract topics
    NUM_TOPICS = 6  # Number of topics to extract
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10,
                                                alpha=[0.01] * NUM_TOPICS)
    lda_model.save('./data/model5.gensim')

    topics = lda_model.print_topics(num_words=7)

    # Print words associated with latent topics to console
    print('\n' * 5)
    for i, topic in enumerate(topics):
        print('Topic %d: %s' % (i, topic))
