"""
This module implements helper functions providing ancillary functionality to other modules and functions.
"""

import datetime
import json
import sys
import time
from multiprocessing import current_process
from bs4 import BeautifulSoup
from bson.json_util import dumps

import DataCollection
from config import ROOT_DIR
import os
import re
from math import floor
from polyglot.text import Text
import pycountry as country
import pandas as pd


def print_progress(iteration, total, prefix='', prog='', round_avg=0, suffix='', time_lapsed=0.0, decimals=1,
                   bar_length=100):
    """
    Creates terminal progress bar by being called in a loop.

    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: positive number of decimals in percent complete (Int)
    :param bar_length: character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = u'\u258B' * filled_length + '-' * (bar_length - filled_length)
    pending_time = (time_lapsed / iteration) * (total - iteration)
    minutes = int(pending_time / 60)
    seconds = round(pending_time % 60)
    suffix = '%d mins, %g secs remaining' % (minutes, seconds)
    sys.stdout.write(
        '\r%s |%s| %s%s - Request %d of %d - %s - %s - %s' % (
            prefix, bar, percents, '%', iteration, total, current_process().name, prog, suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def identify_language(text):
    """
    Identifies language from string using polyglot package.

    :param text: String to use for language identification
    :return: Language name (English)
    """

    try:
        if text is not ('' or None):
            text = Text(text)
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

    return language_name


def split_time_interval(start, end, intv, n_days):
    """
    Splits time interval into chunks according to number of sub-intervals specified as intv.
    Yields iterable of start/end tuples.

    :param start: Start date of time interval
    :param end: End date of time interval
    :param intv: Number of chunks to divide time interval into
    """

    if n_days > 1:
        n_days_micro = floor(n_days / intv)  # Micro search time period length in days
        time_delta = datetime.timedelta(days=n_days_micro - 1)
        # Yield start and end date for time period
        for i in range(intv - 1):
            yield (start + (time_delta + datetime.timedelta(days=1)) * i,
                   start + time_delta + (time_delta + datetime.timedelta(days=1)) * i)
        # Yield last time period
        yield (start + (time_delta + datetime.timedelta(days=1)) * (intv - 1), end)

    else:
        n_days_micro = n_days
        time_delta = end - start
        yield (start, end)

    print('Time delta per time frame: %s:' % time_delta)
    print('Days per time frame: %d' % n_days_micro)


def check_access_tokens(token_index, response):
    """
    Checks state of access tokens and prints state in console; Pauses calling thread if limit is sufficiently low.

    :param token_index: Index of token currently in use
    :param response: Response object from last API request
    :return:
    """

    sleep_threshold = 12
    min_sleep_time = 5
    max_sleep_time = 30
    try:
        print('\n\nRemaining/Limit for token %d: %d/%d' % (token_index,
                                                           int(response.headers['X-RateLimit-Remaining']),
                                                           int(response.headers['X-RateLimit-Limit'])))
        remaining_requests = int(response.headers['X-RateLimit-Remaining'])
        if remaining_requests <= sleep_threshold:
            sleep_duration = min_sleep_time + (sleep_threshold - remaining_requests) * (
                    (max_sleep_time - min_sleep_time) / sleep_threshold)
            time.sleep(sleep_duration)
            print('Execution paused for %s seconds.' % round(sleep_duration, 2))
        else:
            pass
    except KeyError as e:
        print('\nError retrieving X-RateLimit for token %d: %s\n' % (token_index, e.args))


def get_access_tokens():
    """
    Retrieves GitHub Search API authentication tokens from files.

    :return: List of token lists
    """
    # Specify path to credentials
    cred_path = os.path.join(ROOT_DIR, 'DataCollection/credentials')
    # List files with GitHub Access Tokens
    files = [os.path.join(cred_path, f) for f in os.listdir(cred_path) if
             re.match(r'GitHub_Access_Token.*', f)]

    # Initialize empty list of token lists
    token_lists = []

    # Load tokens from text files
    for file in files:
        with open(file, 'r') as f:
            access_tokens = [token.rstrip('\n') for token in f.readlines()]
        # Add token list to list of token lists
        token_lists.append(access_tokens)

    return token_lists


def extract_from_readme(response):
    """
    Gets plain text from readme file.

    :param response: Res
    :return: Plain plain_text, link_list and reference_list of readme file
    """
    plain_text = None
    link_list = []
    reference_list = []
    # Request successful
    if response.status_code == 200:

        # Extract plain_text and remove all line breaks
        soup = BeautifulSoup(response.text, features='lxml')

        # Find all arxiv links and append to reference_list
        journals = ['arxiv', 'ieee', 'researchgate', 'acm.org']
        for reference in soup.findAll('a', attrs={'href': re.compile('(' + '|'.join(journals) + ')')}):
            reference_list.append(reference.get('href'))

        # Find all links and append to link_list
        for link in soup.findAll('a', attrs={'href': re.compile('^http://')}):
            link_list.append(link.get('href'))

        # Remove references from link_list
        link_list = list(set(link_list) - set(reference_list))

        plain_text = ''.join(soup.find_all(text=True))
        plain_text = plain_text.replace('\n', ' ').replace('\t', ' ').replace('\r', '')

        # Set plain_text to null if empty string
        if plain_text == '':
            plain_text = None

    # Request unsuccessful
    elif response.status_code == 404:
        pass
    # print(' - Repository without readme found for: %s\n%s' % (response.plain_text, response.request.url))

    elif response.status_code == 403:
        print('Access denied.')

        # print('\n\nLimit: %d' % int(response.headers['X-RateLimit-Limit']))
        # print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))

    else:
        print('Unknown error occurred while parsing readme file: %d' % response.status_code)
        print(response.reason)

    return plain_text, link_list, reference_list


def get_data_from_collection(path_to_data, collection_name):
    """
    Retrieves data from database collection and store locally to file.

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


def get_df_from_json(file_path):
    """
    Loads data from json file and returns it as a pandas data frame.

    :param file_path: File path to json file within 'data' folder
    :return: Pandas data frame
    """
    # Specify path to saved repository data
    path_to_data = os.path.join(ROOT_DIR, 'DataCollection/data/', file_path)

    # JOB: Load json data as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # JOB: Make DataFrame from json
    data_frame = pd.DataFrame(data)

    return data_frame


def load_data_to_df(file_path, download_data=False):
    """
    Loads data from json file specified in path and returns as pandas data frame.
    If download_data flag is true, the data is loaded from data base first.

    :param file_path: Path to json file
    :param download_data: Flag indicating whether or not to retrieve data from database
    :return: Data frame containing data from json file
    """

    # Specify path to saved repository data
    path_to_data = os.path.join(ROOT_DIR, file_path)

    if download_data:
        # Download data from database (only perform when data changed)
        get_data_from_collection(path_to_data, 'Repos_New')

    # JOB: Load json data as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # JOB: Make DataFrame from json
    data_frame = pd.DataFrame(data)

    return data_frame


def get_layer_type_name(layer_type):
    """
    Transforms keras layer type name into full name
    :param layer_type: Keras layer type name
    :return: Full layer type name
    """

    translation_dict = {
        'Dense': 'Dense Layer',
        'Dropout': 'Dropout Layer',
        'Flatten': 'Flattening Layer',
        'SpatialDropout1D': 'Spatial 1D Dropout Layer',
        'SpatialDropout2D': 'Spatial 2D Dropout Layer',
        'SpatialDropout3D': 'Spatial 3D Dropout Layer',
        'Conv1D': '1D Convolutional Layer',
        'Conv2D': '2D Convolutional Layer',
        'SeparableConv1D': 'Depthwise Separable 1D Convolution',
        'SeparableConv2D': 'Depthwise Separable 2D Convolution',
        'DepthwiseConv2D': 'Depthwise Separable 2D Convolution',
        'Conv2DTranspose': 'Transposed Convolution Layer',
        'Conv3D': '3D Convolution Layer',
        'Conv3DTranspose': 'Transposed Convolution Layer',
        'Cropping1D': 'Cropping Layer for 1D Input',
        'Cropping2D': 'Cropping Layer for 2D Input',
        'Cropping3D': 'Cropping Layer for 3D Data',
        'UpSampling1D': 'Upsampling Layer for 1D Inputs',
        'UpSampling2D': 'Upsampling Layer for 2D Inputs',
        'UpSampling3D': 'Upsampling Layer for 3D Inputs',
        'ZeroPadding1D': 'Zero-Padding Layer for 1D Input',
        'ZeroPadding2D': 'Zero-Padding Layer for 2D Input',
        'ZeroPadding3D': 'Zero-Padding Layer for 3D Input',
        'MaxPooling1D': 'Max Pooling Layer',
        'MaxPooling2D': 'Max Pooling 2D Layer',
        'MaxPooling3D': 'Max Pooling 3D Layer',
        'AveragePooling1D': 'Average Pooling Layer',
        'AveragePooling2D': 'Average Pooling 2D Layer',
        'AveragePooling3D': 'Average Pooling 3D Layer',
        'GlobalMaxPooling1D': 'Global Max Pooling',
        'GlobalMaxPooling2D': 'Global Max 2D Pooling',
        'GlobalMaxPooling3D': 'Global Max 3D Pooling',
        'GlobalAveragePooling1D': 'Global Average Pooling',
        'GlobalAveragePooling2D': 'Global Average 2D Pooling',
        'GlobalAveragePooling3D': 'Global Average 3D Pooling',
        'LocallyConnected1D': 'Locally-Connected Layer',
        'LocallyConnected2D': 'Locally-Connected 2D Layer',
        'RNN': 'Recurrent Layer',
        'SimpleRNN': 'Fully-Connected Recurrent Layer',
        'GRU': 'Gated Recurrent Unit',
        'LSTM': 'Long Short-Term Memory Layer',
        'ConvLSTM2D': 'Convolutional LSTM',
        'ConvLSTM2DCell': 'Cell Class for the ConvLSTM2D Layer',
        'SimpleRNNCell': 'Cell Class for SimpleRNN',
        'GRUCell': 'Cell Class for GRU Layer',
        'LSTMCell': 'Cell Class for LSTM Layer',
        'CuDNNGRU': 'Fast GRU Implementation',
        'CuDNNLSTM': 'Fast LSTM Implementation',
        'Embedding': 'Embedding Layer',
        'BatchNormalization': 'Batch Normalization Layer',
        'GaussianNoise': 'Gaussian Additive Zero-Centered Noise',
        'GaussianDropout': 'Gaussian Dropout Layer',
        'AlphaDropout': 'Alpha Dropout Layer',
    }

    return_name = translation_dict.get(layer_type, layer_type.capitalize())

    return return_name


def get_loss_function_name(function_name):
    """
    Transforms keras loss function name into full loss function name
    :param function_name: Keras loss function name to convert
    :return: Full loss function name
    """

    translation_dict = {
        'mean_squared_error': 'Mean Squared Error',
        'mse': 'Mean Squared Error',
        'mean_absolute_error': 'Mean Absolute Error',
        'mae': 'Mean Absolute Error',
        'mean_absolute_percentage_error': 'Mean Absolute Percentage Error',
        'mean_squared_logarithmic_error': 'Mean Squared Logarithmic Error',
        'squared_hinge': 'Squared Hinge',
        'hinge': 'Hinge',
        'categorical_hinge': 'Categorial Hinge',
        'logcosh': 'Logarith of Hyperbolic Cosine of the Prediction Error',
        'categorical_crossentropy': 'Categorical Cross Entropy',
        'sparse_categorical_crossentropy': 'Sparse Categorical Cross Entropy',
        'binary_crossentropy': 'Binary Cross Entropy',
        'kullback_leibler_divergence': 'Kullback-Leibler Divergence',
        'poisson': 'Poisson',
        'cosine_proximity': 'Cosine Proximity',
    }

    return_name = translation_dict.get(function_name, function_name.capitalize())

    return return_name


def get_optimizer_name(optimizer):
    """
    Transforms keras optimizer name into full optimizer name

    :param optimizer: Keras optimizer name
    :return: Full optimizer name
    """

    translation_dict = {
        'SGD': 'Stochastic Gradient Descent Optimizer',
        'RMSprop': 'RMSProp Optimizer',
        'Adagrad': 'Adagrad Optimizer',
        'Adadelta': 'Adadelta Optimizer',
        'Adam': 'Adam Optimizer',
        'Adamax': 'Adamax Optimizer',
        'Nadam': 'Nesterov Adam Optimizer'
    }

    return_name = translation_dict.get(optimizer, optimizer.capitalize())

    return return_name


def filter_data_frame(data_frame, has_architecture=False, has_english_readme=False, no_behavioral_cloning=False,
                      long_readme_only=False, min_length=1000, reset_index=True):
    """
    Filters repository DataFrame by specified attributes.

    :param data_frame: DataFrame containing the data to be filtered
    :param has_architecture: Flag indicating whether to filter by availability of architecture information
    :param has_english_readme: Flag indicating whether to filter by readme language being English
    :param no_behavioral_cloning: Flag indicating whether to exclude behavior cloning repositories
    :param long_readme_only: Flag indicating whether to only include repositories with a minimum readme length as
    specified with min_length attribute
    :param min_length: Attribute specifying minimum length if long_readme_only flag is set to true
    :return: Filtered DataFrame
    """

    # JOB: Filter by repositories with architecture information
    if has_architecture:
        data_frame = data_frame[(data_frame['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))) | (
            data_frame['py_data'].apply(func=lambda x: x.get('model_file_found')))]

    # JOB: Filter by repositories with non-empty English readme
    if has_english_readme:
        data_frame = data_frame[
            (data_frame['readme_language'] == 'English') & (data_frame['readme_text'] != None)]

    # JOB: Filter out repositories with allusion to behavioral cloning udaciy project in name
    if no_behavioral_cloning:
        data_frame = data_frame[
            ~(data_frame['repo_name'].str.contains('([Bb]ehavior|[Bb]ehaviour|[Cc]ar|[Cc]lon|[Dd]riv)'))]

    # JOB: Filter by repositories with specified minimum readme length
    if long_readme_only:
        data_frame = data_frame[data_frame['readme_text'].str.len() > min_length]

    if reset_index:
        # JOB: Reset index
        data_frame.reset_index(inplace=True, drop=True)  # Reset index

    return data_frame
