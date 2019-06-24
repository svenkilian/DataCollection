# Standalone executable module implementing experimental NLP techniques to textual information.

from collections import Counter
from urllib.parse import urlparse

import gensim
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm

from Classifiers.AttributeClfs import train_test_doc2vec_nn_application, get_nn_type_from_architecture, \
    nn_application_encoding, train_test_nn_application, train_test_nn_type
from HelperFunctions import load_data_to_df

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


def calculate_similarities(data_frame):
    """
    Calculates similarities between repositories.

    :param data_frame: Data frame containing readme texts.
    :return:
    """

    # JOB: Filter repositories by repositories with non-empty English readmes with minimum length of 5000 characters
    data_frame = data_frame[(data_frame['readme_language'] == 'English') & (data_frame['readme_text'] != None) & (
            (data_frame['readme_text'].str.len()) > 5000) & ~(
        data_frame['repo_name'].str.contains(
            '([Bb]ehavior|[Bb]ehaviour|[Cc]ar|[Cc]lon|[Dd]riv)'))].head(1000)
    data_frame.reset_index(inplace=True)  # Reset index

    # JOB: Load pre-trained Doc2Vec model
    print('Loading pre-trained model ...')
    doc2vec_model = Doc2Vec.load(get_tmpfile('doc2vec_model'))

    # JOB: Pair-wise comparison between repositories based on readme texts using cosine similarity
    for ix_outer, repo in tqdm(data_frame.iterrows(), total=data_frame.shape[0]):
        for ix_inner in range(ix_outer + 1, data_frame.shape[0]):
            vec_1 = doc2vec_model.infer_vector(gensim.utils.simple_preprocess(repo['readme_text']))
            vec_2 = doc2vec_model.infer_vector(
                gensim.utils.simple_preprocess(data_frame.loc[[ix_inner], 'readme_text'].values[0]))

            # Calculate cosine similarity
            similarity = cosine_similarity(vec_1.reshape(1, -1), vec_2.reshape(1, -1))[0, 0]

            # Only consider strong similarities
            if 0.8 < similarity < 0.9:
                print('(%d, %d): %g' % (ix_outer, ix_inner, round(similarity, 2)))
                print(tabulate(data_frame.loc[[ix_outer, ix_inner], ['repo_url']], tablefmt='psql', showindex=True,
                               headers='keys'))


if __name__ == '__main__':
    """
    Main method
    """

    # JOB: Load data from file
    print('Loading data ...')
    data_frame = load_data_to_df('DataCollection/data/data.json', download_data=False)
    train_test_nn_type(data_frame, write_to_db=True)
    # train_test_nn_application(data_frame, write_to_db=False)
    # model = train_test_doc2vec_nn_application(data_frame, 'nn_type', get_nn_type_from_architecture)
    # model = train_test_doc2vec_nn_application(data_frame, 'nn_application', nn_application_encoding)
    # Calculate repository similarities based on readme texts
    # calculate_similarities(data_frame)
