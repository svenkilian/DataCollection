import gensim
import nltk
import spacy
from spacy.lang.en import English
from tqdm import tqdm

from NLP_Functions import prepare_text_for_lda
from gensim import corpora
import pickle
from config import ROOT_DIR
import os
import random
import requests
import pymongo
import pandas as pd
import numpy as np
import json
from bson.json_util import dumps, loads
# import matplotlib.pyplot as plt
# import seaborn as sns


from polyglot.text import Text
import pycountry as country
import seaborn as sns

if __name__ == '__main__':
    # Retrieve database credentials
    # print(ROOT_DIR)
    # cred_path = os.path.join(ROOT_DIR, 'DataCollection/credentials/connection_creds.txt')
    path_to_data = os.path.join(ROOT_DIR, 'DataCollection/data/data.json')
    #
    # with open(cred_path, 'r') as f:
    #     connection_string = f.read()
    #
    # # Establish database connection
    # client = pymongo.MongoClient(connection_string)
    # collection = client.GitHub.Repos_Exp
    # print(client.server_info())
    #
    # # Save database query result to json
    # data = dumps(collection.find({}))
    # # print(data)
    # with open(path_to_data, 'w') as file:
    #     file.write(data)

    # Load json as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # Make DataFrame from json
    data_frame = pd.DataFrame(data)
    print(data_frame[['repo_full_name', 'readme_text']].iloc[20:30])

    access_path = os.path.join(ROOT_DIR, 'DataCollection/credentials/GitHub_Access_Token.txt')

    # Print column names
    print(data_frame.columns)

    # data_frame['year'] = data_frame['repo_created_at'].dt.year

    for index, row in tqdm(data_frame.iterrows(), total=data_frame.shape[0]):
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

        data_frame.at[index, 'language_readme'] = language_name
        data_frame.at[index, 'language_readme_code'] = language_code

    print(data_frame[data_frame['language_readme'] != 'English'].loc[:, ['readme_text', 'language_readme']])

    data_frame['language_readme'].astype('category')

    print('\n\nUnique languages:')
    print(data_frame['language_readme'].unique())

    # Language distribution
    print('\n\n')
    print(data_frame.groupby(['language_readme']).size().reset_index(name='Count').sort_values(
        'Count')[-10:])

    # Filter by English readmes
    data_en = data_frame[data_frame['language_readme'] == 'English'][:]

    data_en.to_excel(r'Output.xlsx', engine='xlsxwriter')

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
    NUM_TOPICS = 10  # Number of topics to extract
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10,
                                                alpha=[0.01] * NUM_TOPICS)
    lda_model.save('./data/model5.gensim')

    topics = lda_model.print_topics(num_words=7)

    print('\n' * 5)
    for i, topic in enumerate(topics):
        print('Topic %d: %s' % (i, topic))

    # repo_count_per_year = data_frame.groupby(['year']).size().reset_index(name='counts of repos')
    # print(repo_count_per_year)
    #
    # has_wiki_count = data_frame.groupby(['has_wiki']).size().reset_index(name='counts of repos')
    # print(has_wiki_count)
    #
    # #has_homepage_count = data_frame.groupby(['homepage']).size().reset_index(name='counts of repos')
    # has_homepage_count = data_frame.iloc[:,3].isna().sum()
    # print(has_homepage_count)
    #
    # owner_count = data_frame.groupby(['repo_owner']).size().reset_index(name='counts of repos').sort_values(by=['counts of repos'], ascending=False)
    # print(owner_count)
    #
    # print(owner_count[owner_count['counts of repos'] > 1].sum())
    #
    # #corr = data_frame.loc[:, data_frame.dtypes == 'int64'].corr()
    # #sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
    #             cmap=sns.diverging_palette(220, 10, as_cmap=True))
    # #plt.show()
