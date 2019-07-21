# Standalone executable module implementing experimental NLP techniques to textual information.
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import multiprocessing
import os
import re
import sys
import time
from collections import Counter
from itertools import repeat
from urllib.parse import urlparse

import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from gensim.models import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from Classifiers.AttributeClfs import train_test_doc2vec_nn_application, get_nn_type_from_architecture, \
    nn_application_encoding, train_test_nn_application, train_test_nn_type
from HelperFunctions import load_data_to_df, filter_data_frame
from NLPFunctions import read_corpus, perform_doc2vec_embedding
from scipy.stats import rankdata
from config import ROOT_DIR
from multiprocessing import Pool

pd.set_option('mode.chained_assignment', None)


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
    data_frame['has_readme'] = data_frame.apply(func=lambda row: (row['readme_text'] != None) & (
            row['readme_text'] != ''), axis=1)
    data_frame['has_english_readme'] = data_frame.apply(
        func=lambda row: (row['readme_language'] == 'English') & (row['readme_text'] != None) & (
                row['readme_text'] != ''), axis=1)

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

    # Readme analysis
    readme_count = data_frame.groupby(['has_readme']).size().reset_index(name='Counts of repos').set_index(
        'has_readme')
    print(readme_count)
    print(2 * '\n')

    english_readme_count = data_frame.groupby(['has_english_readme']).size().reset_index(
        name='Counts of repos').set_index(
        'has_english_readme')
    print(english_readme_count)
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
    extracted_architecture_count = data_frame.groupby(['extracted_architecture']).size().reset_index(
        name='Counts of repos').set_index('extracted_architecture')
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
    print(tabulate(data_frame.sample(20)['all_links'], tablefmt='psql', showindex=True, headers='keys'))

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
                   'keras-neural-networks', 'flask', 'mnist', 'numpy', 'scikit-learn', 'keras-models', 'kaggle',
                   'theano', 'data-science', 'docker', 'artificial-neural-networks', 'dataset', 'machinelearning',
                   'deeplearning', 'artificial-intelligence', 'ai', 'machine-learning-algorithms',
                   'behavioral-cloning'}
    type_list = {'convolutional-neural-networks', 'cnn', 'lstm', 'rnn', 'cnn-keras', 'recurrent-neural-networks',
                 'lstm-neural-networks', 'generative-adversarial-network', 'gan', 'reinforcement-learning',
                 'deep-reinforcement-learning', 'autoencoder'}

    # Filter by repositories with topic label from type_list
    df = df[~df['Tag'].isin(type_list.union(stop_topics))].head(50)

    # Print new data frame
    print(df.head(60))

    return df['Tag'].tolist()


def display_pca_scatterplot(doc_vecs, labels, words=None, sample=0):
    """
    Show scatterplot of dimensionally reduced data.

    :param model: Trained gensim embedding model
    :param words:
    :param sample: Sample size
    :return:
    """

    # JOB: 2-dimension representation
    reduced = PCA(n_components=2).fit_transform(doc_vecs)[:, :2]

    fig_1 = plt.figure(figsize=(6, 6))
    fig_1.gca().scatter(reduced[:, 0], reduced[:, 1], edgecolor=None, c=labels, cmap=plt.cm.get_cmap('RdBu'), s=10)

    plt.show()

    # JOB: 3-dimensional representation
    reduced = PCA(n_components=3).fit_transform(doc_vecs)[:, :3]

    fig = plt.figure(figsize=(16, 12))
    ax = Axes3D(fig)
    scatter = ax.scatter(reduced[:, 0],
                         reduced[:, 1],
                         reduced[:, 2],
                         c=labels,
                         cmap=plt.cm.get_cmap('Dark2'),
                         s=15,
                         edgecolor='none',
                         marker='o')
    # plt.title("Semantic Tf-Idf-SVD reduced plot of Sincere-Insincere data distribution")
    plt.xlabel("First dimension")
    plt.ylabel("Second dimension")
    plt.legend()
    plt.xlim(-6, 6)
    plt.ylim(-7.5, 5)
    plt.gca().set_zlim(-4, 5)
    plt.colorbar(scatter)
    plt.show()


def visualize_doc2vec_embedding(data_frame):
    """
    Calculates similarities between repositories based on their readme file's doc2vec representation.

    :param data_frame: Data frame containing readme texts.
    :return:
    """

    data_frame = data_frame[data_frame['application'].notna()]

    # data_frame['nn_type'] = data_frame['nn_type'].apply(
    #     lambda x: 1 if x[0] == 'recurrent_type' else 2 if x[0] == 'conv_type' else 0)
    # data_frame = data_frame[data_frame['nn_type'] != 0]

    data_frame['application'] = data_frame['application'].apply(
        lambda x: 1 if x[0] == 'nlp' else 2 if x[0] == 'images' else 0)

    labels = data_frame['application']

    # Get repo name array
    repo_names_index = data_frame.index.tolist()

    # JOB: Load pre-trained Doc2Vec model
    print('Loading pre-trained model ...')
    doc2vec_model = Doc2Vec.load(os.path.join(ROOT_DIR, 'DataCollection/data/doc2vec_model'))

    doc_vecs = [doc2vec_model.docvecs[name] for name in repo_names_index]

    print(len(doc_vecs))
    print(type(doc_vecs))

    display_pca_scatterplot(doc_vecs, labels)


def calculate_similarity_row_vector(vectorized_readmes, row_index):
    # print('Current row index being processed: %d' % row_index)
    similarity_row = np.zeros(len(vectorized_readmes))

    for col_index in range(row_index + 1, vectorized_readmes.shape[0]):
        vec_1 = vectorized_readmes[row_index]
        vec_2 = vectorized_readmes[col_index]

        # Calculate cosine similarity
        similarity = cosine_similarity(vec_1.reshape(1, -1), vec_2.reshape(1, -1))[0, 0]

        similarity_row[col_index] = similarity

    return similarity_row


def calculate_similarities(data_frame):
    """
    Calculates similarities between repositories based on their readme file's doc2vec representation.

    :param data_frame: Data frame containing readme texts.
    :return:
    """

    data_frame.reset_index(inplace=True, drop=True)

    # Get repo name array
    repo_names_index = data_frame['repo_full_name'].values

    # Initialize empty numpy array
    # similarity_array = np.zeros((data_frame.shape[0], data_frame.shape[0]))

    # JOB: Load pre-trained Doc2Vec model
    print('Loading pre-trained model ...')
    doc2vec_model = Doc2Vec.load(os.path.join(ROOT_DIR, 'DataCollection/data/doc2vec_model'))

    # Preprocess readmes
    print('Pre-processing readmes ...')
    preprocessed_readmes = list(
        read_corpus(data_frame['repo_full_name', 'readme_text'].set_index(['repo_full_name'], drop=True, inplace=True),
                    tokens_only=True))

    # Initialize empty array
    vectorized_readmes = np.empty(
        (len(preprocessed_readmes), doc2vec_model.vector_size))

    print('Vectorizing readmes ... \n')
    for i, row in tqdm(enumerate(vectorized_readmes), total=len(vectorized_readmes)):
        vectorized_readmes[i] = doc2vec_model.infer_vector(preprocessed_readmes[i])

    # Load vectorized readmes from saved model
    # vectorized_readmes = doc2vec_model.docvecs

    start_time = time.time()
    print('Calculating similarities ...')

    process_pool = Pool(processes=multiprocessing.cpu_count())
    row_indices = range(vectorized_readmes.shape[0])

    similarity_array = np.asarray(
        process_pool.starmap(calculate_similarity_row_vector, zip(repeat(vectorized_readmes), row_indices)))

    process_pool.close()
    process_pool.join()

    # print('Calculating similarities ...')
    # # JOB: Pair-wise comparison between repositories based on readme texts using cosine similarity
    # for ix_outer, repo in tqdm(data_frame.iterrows(), total=data_frame.shape[0]):
    #     for ix_inner in range(ix_outer + 1, data_frame.shape[0]):
    #         vec_1 = vectorized_readmes[ix_outer]
    #         vec_2 = vectorized_readmes[ix_inner]
    #         # vec_1 = doc2vec_model.infer_vector(gensim.utils.simple_preprocess(repo['readme_text']))
    #         # vec_2 = doc2vec_model.infer_vector(
    #         #     gensim.utils.simple_preprocess(data_frame.loc[[ix_inner], 'readme_text'].values[0]))
    #
    #         # Calculate cosine similarity
    #         similarity = cosine_similarity(vec_1.reshape(1, -1), vec_2.reshape(1, -1))[0, 0]
    #         similarity_array[ix_outer, ix_inner] = similarity
    #
    #         # search_words = re.compile(u'([Bb]ehavior|[Bb]ehaviour|[Cc]ar|[Cc]lon|[Dd]riv)')
    #         #
    #         # # Only consider strong similarities for console output
    #         # if len(repo['readme_text']) > 1000:
    #         #     if not search_words.search(repo['repo_name']):
    #         #         if 0.9 < similarity < 1.0:
    #         #             print('(%d, %d): %g' % (ix_outer, ix_inner, round(similarity, 2)))
    #         #             print(tabulate(data_frame.loc[[ix_outer, ix_inner], ['repo_url']], tablefmt='psql', showindex=True,
    #         #                            headers='keys'))

    end_time = time.time()

    print('Execution time: %g seconds.' % round(end_time - start_time, 2))

    # Make matrix symmetric
    print('Adding lower triangle to similarity matrix ...')
    for i in range(1, len(similarity_array)):
        for j in range(0, i):
            similarity_array[i][j] = similarity_array[j][i]

    np.fill_diagonal(similarity_array, np.nan)
    # print(similarity_array)
    # print(similarity_array.max())
    # print(similarity_array.min())

    # Array to pandas dataframe with column and row indices
    similarity_df = pd.DataFrame(similarity_array, index=repo_names_index, columns=repo_names_index)

    # print(tabulate(similarity_df, headers='keys',
    #                tablefmt='psql', showindex=True))

    ranks_array = np.empty_like(similarity_array)

    for i, row in enumerate(similarity_array):
        ranks_array[i] = (len(similarity_array)) - rankdata(row, method='ordinal')

    similarity_ranks_df = pd.DataFrame(ranks_array, index=repo_names_index, columns=repo_names_index)

    # print(tabulate(similarity_ranks_df, headers='keys',
    #                tablefmt='psql', showindex=True))

    # Save similarity scores data frame to json
    print('Saving similarity scores ...')
    output_file = os.path.join(ROOT_DIR, 'DataCollection/data/filtered_data_sims.json')  # Specify output name
    # output_file_tbl = os.path.join(ROOT_DIR, 'DataCollection/data/filtered_data_sims.csv')  # Specify output name
    similarity_df.to_json(output_file)
    # similarity_df.to_csv(output_file_tbl)

    # Save similarity rank data frame to json
    print('Saving similarity ranks ...')
    output_file = os.path.join(ROOT_DIR, 'DataCollection/data/filtered_data_sims_ranks.json')  # Specify output name
    similarity_ranks_df.to_json(output_file)


def compare_group_similarity(data_frame):
    readme_scores = pd.read_json(os.path.join(ROOT_DIR, 'DataCollection/data/filtered_data_sims.json'))

    # NLP repo indices
    nlp_repos = data_frame['application'].apply(lambda s: 'nlp' in s if not isinstance(s, float) else False)
    nlp_repos_index = nlp_repos[nlp_repos].index.tolist()

    nlp_repos_index = list(set(nlp_repos_index).intersection(set(readme_scores.index.tolist())))

    # Image repo indices
    images_repos = data_frame['application'].apply(lambda s: 'images' in s if not isinstance(s, float) else False)
    images_repos_index = images_repos[images_repos].index.tolist()

    images_repos_index = list(set(images_repos_index).intersection(set(readme_scores.index.tolist())))

    within_images = sum(readme_scores.loc[images_repos_index, images_repos_index].sum(skipna=True, axis=0)) / ((
            len(images_repos_index) ** 2 - len(images_repos_index)))

    within_nlp = sum(readme_scores.loc[nlp_repos_index, nlp_repos_index].sum(skipna=True, axis=0)) / ((
            len(nlp_repos_index) ** 2 - len(nlp_repos_index)))

    between = sum(readme_scores.loc[images_repos_index, nlp_repos_index].sum(skipna=True, axis=0)) / (
            len(images_repos_index) * len(nlp_repos_index) - max(len(images_repos_index), len(nlp_repos_index)))

    print('Within images similarity: %g' % round(within_images, 3))
    print('Within nlp similarity: %g' % round(within_nlp, 3))
    print('Between group similarity: %g' % round(between, 3))

    weighted = (len(nlp_repos_index) * within_nlp + len(images_repos_index) * within_images) / (
                len(nlp_repos_index) + len(
            images_repos_index))

    print(weighted)

    ratio = weighted / between

    print(ratio)


if __name__ == '__main__':
    """
    Main method
    """

    # JOB: Load data from file
    print('Loading data ...')
    full_readme_data = filter_data_frame(load_data_to_df('DataCollection/data/data.json', download_data=False),
                                         has_english_readme=True, long_readme_only=True, min_length=200)

    full_readme_data = full_readme_data.set_index(['repo_full_name'], drop=True)

    data_frame = load_data_to_df('DataCollection/data/filtered_data.json', download_data=False)

    # Filter data
    # data_frame = filter_data_frame(data_frame, has_architecture=False, has_english_readme=True)
    # print(tabulate(data_frame.head(20), headers='keys',
    #                tablefmt='psql', showindex=True))
    # get_stats(data_frame)

    train_test_nn_type(full_readme_data, write_to_db=False, set_index=False, load_data=True)
    # train_test_nn_application(full_readme_data, write_to_db=False)
    # model = train_test_doc2vec_nn_application(full_readme_data, 'nn_type', get_nn_type_from_architecture)
    # model = train_test_doc2vec_nn_application(data_frame, 'nn_application', nn_application_encoding)

    # JOB: Perform Doc2Vec embedding
    # print('Performing doc2vec ...')
    # model = perform_doc2vec_embedding(full_readme_data['readme_text'], train_model=True, vector_size=100, epochs=20)

    # Calculate repository similarities based on readme texts

    # calculate_similarities(data_frame)

    # Visualize embedding

    # data_frame = data_frame.set_index(['repo_full_name'], drop=True)
    # visualize_doc2vec_embedding(full_readme_data)

    # compare_group_similarity(full_readme_data)
