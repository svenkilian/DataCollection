import os
import re
import time
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain
from HelperFunctions import print_progress
from NLPFunctions import LemmaTokenizer
from joblib import dump, load
from tabulate import tabulate

from config import ROOT_DIR


def nn_type_encoding(df, type_list):
    """
    Performs multi-label binary encoding for neural network type and filters for repositories containing at
    least one of the extracted labels.
    :param df: Data frame containing text data to use for encoding
    :param type_list: List of neural network types
    :return: Filtered data frame, list of neural network types for classification
    """

    # Dict matching identical labels
    type_match_dict = {'convolutional-neural-networks': 'cnn',
                       'cnn-keras': 'cnn',
                       'cnns': 'cnn',
                       re.compile('^.*cnn.*$'): 'cnn',
                       re.compile('^.*convolution.*$'): 'cnn',
                       'recurrent-neural-networks': 'rnn',
                       'generative-adversarial-network': 'gan',
                       'deep-reinforcement-learning': 'reinforcement-learning',
                       'lstm-neural-networks': 'cnn',
                       'lstm': 'cnn',
                       'long-short-term-memory-models': 'cnn',
                       'long-short-term-memory': 'cnn',
                       }

    # Deduplicate type_list
    type_list = [item for item in type_list if item not in type_match_dict.keys()]

    # Replace synonymous types
    df['repo_tags'] = df['repo_tags'].apply(
        lambda topic_list: [type_match_dict.get(topic, topic) for topic in
                            topic_list] if topic_list is not None else [])

    # Set multi-label values
    for nn_type in type_list:
        df[nn_type] = pd.Series(
            [1 if nn_type in (topic_list if topic_list is not None else []) else 0 for topic_list in
             df['repo_tags'].tolist()])

    # Filter data for repositories containing at least one of the type labels
    df = df[df[type_list].any(axis=1)]

    return df, type_list


def nn_application_encoding(df, type_list):
    """
    Performs encoding for neural network application type and filters for repositories containing at
    least one of the extracted labels.
    :param df: Data frame containing text data to use for encoding
    :param type_list: List of neural network applications
    :return: Filtered data frame, list of neural network applications for classification
    """

    # Dict matching identical labels
    type_match_dict = {'natural-language-processing': 'nlp',
                       'text-classification': 'nlp',
                       'sentiment-analysis': 'nlp',
                       'nlp-machine-learning': 'nlp',
                       'seq2seq': 'nlp',
                       'word2vec': 'nlp',
                       'word-embeddings': 'nlp',
                       'text-mining': 'nlp',
                       'text-prediction': 'nlp',
                       'text-generation': 'nlp',
                       'part-of-speech-tagger': 'nlp',
                       'named-entity-recognition': 'nlp',
                       'computer-vision': 'images',
                       'semantic-segmentation': 'images',
                       'image-classification': 'images',
                       'image-processing': 'images',
                       'face-recognition': 'images',
                       'facial-recognition': 'images',
                       'image-recognition': 'images',
                       'face-detection': 'images',
                       'real-time-face-detection': 'images',
                       'image-segmentation': 'images',
                       'resnet': 'images',
                       'vgg16': 'images',
                       'cifar10': 'images',
                       'opencv': 'images',
                       'opencv-python': 'images',
                       'digit-recognition': 'images',
                       'object-detection': 'images',
                       }

    # Replace synonymous types
    df['repo_tags'] = df['repo_tags'].apply(
        lambda topic_list: [type_match_dict.get(topic, topic) for topic in
                            topic_list] if topic_list is not None else [])

    # Set multi-label values
    for application_type in type_list:
        df[application_type] = pd.Series(
            [1 if application_type in topic_list else 0 for topic_list in df['repo_tags'].tolist()])

    # Filter data for repositories containing at least one of the type labels
    df = df[df[type_list].any(axis=1)]

    return df


def get_nn_type_from_architecture(data_frame):
    """
    Analyzes layer information to extract information about network type.
    :return: NN Type labels
    """

    # Define sets of convolutional and recurrent layers
    convolutional_layers = {
        'Conv1D',
        'Conv2D',
        'SeparableConv1D',
        'SeparableConv2D',
        'DepthwiseConv2D',
        'Conv2DTranspose',
        'Conv3D',
        'Conv3DTranspose',
        'Cropping1D',
        'Cropping2D',
        'Cropping3D',
        'UpSampling1D',
        'UpSampling2D',
        'UpSampling3D',
        'ZeroPadding1D',
        'ZeroPadding2D',
        'ZeroPadding3D',
        'MaxPooling1D',
        'MaxPooling2D',
        'MaxPooling3D',
        'AveragePooling1D',
        'AveragePooling2D',
        'AveragePooling3D',
        'GlobalMaxPooling1D',
        'GlobalMaxPooling2D',
        'GlobalMaxPooling3D',
        'GlobalAveragePooling1D',
        'GlobalAveragePooling2D',
        'GlobalAveragePooling3D',
        'LocallyConnected1D',
        'LocallyConnected2D',
    }
    recurrent_layers = {
        'RNN',
        'SimpleRNN',
        'GRU',
        'LSTM',
        'ConvLSTM2D',
        'ConvLSTM2DCell',
        'SimpleRNNCell',
        'GRUCell',
        'LSTMCell',
        'CuDNNGRU',
        'CuDNNLSTM',
    }

    # Initialize empty series for new features
    data_frame['feed_forward_type'] = pd.Series(np.empty((data_frame.shape[0])))
    data_frame['conv_type'] = pd.Series(np.empty((data_frame.shape[0])))
    data_frame['recurrent_type'] = pd.Series(np.empty((data_frame.shape[0])))

    for index, repo in data_frame.iterrows():

        is_conv_nn = 0
        is_recurrent_nn = 0
        is_feed_forward_nn = 0
        layers = []

        if repo['h5_data'].get('extracted_architecture'):
            layers = repo['h5_data'].get('model_layers')
        elif repo['py_data'].get('model_file_found'):
            layers = repo['py_data'].get('model_layers')

        layer_types = set([layer.get('layer_type') for layer in layers.values()])

        if len(layer_types.intersection(convolutional_layers)) > 0:
            is_conv_nn = 1
        if len(layer_types.intersection(recurrent_layers)) > 0:
            is_recurrent_nn = 1

        if not (is_recurrent_nn or is_conv_nn):
            is_feed_forward_nn = 1

        data_frame.loc[[index], ['feed_forward_type', 'conv_type', 'recurrent_type']] = [is_feed_forward_nn, is_conv_nn,
                                                                                         is_recurrent_nn]
        print_progress(index + 1, len(data_frame))

    return data_frame


def preprocess_data(df):
    """
    Applies pre-processing and returns train-test split of data set.
    :param df: Data frame to process
    :return: Train-test split of preprocessed data and fitted count vectorizer
    """

    # Extract features and target matrices
    X = df['readme_text']
    y = df.iloc[:, 1:]

    # Fit count vectorizer and transform features
    begin_time = time.time()
    count_vectorizer = CountVectorizer(strip_accents='unicode', ngram_range=(1, 2),
                                       tokenizer=LemmaTokenizer(), max_features=500).fit(X)
    X = count_vectorizer.transform(X)
    X = TfidfTransformer(use_idf=True).fit_transform(X)
    end_time = time.time()
    print('Preprocessing/Vectorizing duration: %g seconds' % round((end_time - begin_time), 2))

    print('Shape of data after vectorizing:')
    print(X.shape)
    print(y.shape)

    # Obtain train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    return X_train, X_test, y_train, y_test, count_vectorizer


def train_model(X_train, y_train, clf, load_model=False):
    """
    Trains and returns classifier.
    :param load_model:
    :param clf: Classiier to train
    :param X_train: Features of training set
    :param y_train: Labels of training set
    :return: Trained classifier
    """

    if not load_model:
        classifier = ClassifierChain(clf)
        # predictions = cross_val_predict(classifier, X.astype(float), y.astype(float), cv=3, n_jobs=12,
        #                                 verbose=2)
        # cross_val_accuracy = metrics.f1_score(y, predictions, average='samples')
        classifier.fit(X_train.astype(float), y_train.astype(float))
        dump(classifier, os.path.join(ROOT_DIR, 'DataCollection/data/trained_model.joblib'))

    else:
        classifier = load(os.path.join(ROOT_DIR, 'DataCollection/data/trained_model.joblib'))

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
    # history = model.fit(x_train, y,
    #                     class_weight=class_weight,
    #                     epochs=20,
    #                     batch_size=32,
    #                     validation_split=0.1
    # callbacks = callbacks)

    return classifier


def test_model(X_test, y_test, clf):
    """
    Tests model performance on test data and returns test score.
    :param X_test: Test data features
    :param y_test: Test data labels
    :param clf: Trained classifier to test
    :return: Score
    """
    predictions = clf.predict(X_test.astype(float)).toarray()
    score = metrics.f1_score(y_test, predictions, average='weighted')

    return score


def apply_model(clf, count_vectorizer, data_frame, type_names):
    """
    Applies classifier to readme data passed as a pandas series.
    :param type_names:
    :param count_vectorizer:
    :param clf: Trained classifier for network type classification
    :param data_frame: Data frame containing plain text readme strings
    :return: Predictions in data frame
    """

    vectorizer = count_vectorizer
    X = vectorizer.transform(data_frame['readme_text'])
    predictions = pd.DataFrame(clf.predict(X.astype(float)).todense(), columns=type_names)

    result_df = pd.concat(
        [data_frame[['repo_url', 'readme_text']].reset_index(drop=True), predictions.reset_index(drop=True)],
        ignore_index=True, axis=1)

    result_df.set_axis(labels=['URL', 'Readme', *type_names], axis=1, inplace=True)

    return result_df


def train_test_nn_type(data_frame):
    """
    Trains multi-label classifier on neural network types.
    :param data_frame: Data source
    :return:
    """

    program_start_time = time.time()
    # JOB: Filter by repositories with architecture information
    repos = data_frame[(data_frame['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))) | (
        data_frame['py_data'].apply(func=lambda x: x.get('model_file_found')))].reset_index()

    print('Number of repositories with architecture information: %d' % len(repos))

    # JOB: Extract neural network type information from architecture
    print('Extracting labels from architecture ...')
    df = get_nn_type_from_architecture(repos)

    # Specify list of neural network types
    type_list = ['feed_forward_type', 'conv_type', 'recurrent_type']

    # Print counts of neural network types
    # print(df[type_list].sum())

    # Encode neural network types with multi-level binarization and deduplicate type_list
    # df, type_list = nn_type_encoding(data_frame, type_list)

    # print(tabulate(df[['readme_text', 'repo_tags', *type_list]].sample(20), headers='keys',
    #                tablefmt='psql', showindex=True))

    # df['n_classes'] = df[type_list].apply(func=np.sum, axis=1)
    # print(tabulate(
    #     df[['repo_full_name', 'repo_tags', *type_list, 'n_classes']].sort_values(['n_classes'], ascending=False).head(10),
    #     headers='keys',
    #     tablefmt='psql', showindex=True))
    #
    # plt.hist(df['n_classes'])
    # plt.show()

    # Filter by non-empty English-language readme texts
    # JOB: Filter for data with readme

    # Filter data for repositories with non-empty, English readmes
    df_learn = df[(df['readme_language'] == 'English') & (df['readme_text'] != None)][
        ['readme_text', *type_list]]

    # Apply text preprocessing and split into training and test data
    print('Preprocessing data ...')
    X_train, X_test, y_train, y_test, count_vectorizer = preprocess_data(df_learn)

    print('Number of repositories: %d' % df_learn.shape[0])

    # JOB: Train model
    print('Training model ...')
    begin_time = time.time()
    clf = train_model(X_train, y_train, LinearSVC(max_iter=10000), load_model=False)
    end_time = time.time()
    print('Model fit duration: %g seconds' % round((end_time - begin_time), 2))
    # print('Cross validation accuracy: %g' % round(cross_val_accuracy, 2))

    score = test_model(X_test, y_test, clf)

    print('Score: %g' % round(score, 2))

    program_end_time = time.time()

    print('Program execution duration: %g' % round(program_end_time - program_start_time, 2))

    # Filter for test data
    test_data = data_frame[
        (data_frame['readme_language'] == 'English') &
        (data_frame['readme_text'] != '') &
        (data_frame['readme_text'] != None) &
        ~(data_frame['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))) &
        ~(data_frame['py_data'].apply(func=lambda x: x.get('model_file_found')))]

    # print(tabulate(test_data.sample(10), headers='keys',
    #                tablefmt='psql', showindex=True))

    print('Applying model to test data ...')
    begin_time = time.time()
    result_df = apply_model(clf, count_vectorizer, test_data.sample(10), type_list)
    end_time = time.time()
    print('Prediction time: %g seconds' % round((end_time - begin_time), 2))

    print(tabulate(result_df, headers='keys',
                   tablefmt='psql', showindex=True))


def train_test_nn_application(data_frame):
    """
    Train and test multi-label classifier on neural network application.
    :param data_frame: Data frame contraining training and test data consisting
    of readme file as plain text and binary labels indicating affiliation to application category.
    :return:
    """

    program_start_time = time.time()  # Time training and testing duration

    # Specify list of considered labels for classifier
    topic_list = ['nlp', 'images']

    # JOB: Encode and preprocess data
    # Encode class affiliation for specified labels
    df = nn_application_encoding(data_frame, topic_list)

    print('Number of networks with application labels set: %d' % df.shape[0])

    # Filter data by repositories with non-empty English readmes
    df_learn = df[(df['readme_language'] == 'English') & (df['readme_text'] != None)][
        ['readme_text', *topic_list]]

    print('Number of networks with labels and non-empty English readmes: %d' % df_learn.shape[0])

    # Apply text preprocessing and split into training and test data
    print('Preprocessing data ...')
    X_train, X_test, y_train, y_test, count_vectorizer = preprocess_data(df_learn)

    # JOB: Train model
    print('Training model ...')
    begin_time = time.time()
    clf = train_model(X_train, y_train, LinearSVC(max_iter=10000), load_model=False)
    end_time = time.time()
    print('Model fit duration: %g seconds' % round((end_time - begin_time), 2))

    # JOB: Test trained model on test data
    score = test_model(X_test, y_test, clf)

    print('Score: %g' % round(score, 2))

    program_end_time = time.time()
    print('Program execution duration: %g' % round(program_end_time - program_start_time, 2))
