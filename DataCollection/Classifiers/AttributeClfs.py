import os
import re
import time

import joblib
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy import sparse
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from tabulate import tabulate

import DataCollection
from HelperFunctions import print_progress
from NLPFunctions import LemmaTokenizer, perform_doc2vec_embedding, read_corpus
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

    print('Number of repositories with architecture information: %d' % len(df))

    return df


def nn_decoding(df, topic_list, classification_task, drop_columns=False):
    """
    Maps multi-label encoding of application type back to array containing the respective tag.

    :param classification_task: String indicating which classification task to do encoding for
    :param df: Data frame containing label information in one-hot encoded form
    :param topic_list: List of application types to consider
    :param drop_columns: Flag indicating whether to drop the binary feature columns after decoding
    :return: Data frame with additional column containing the decoded labels
    """

    # Initialize column with empty lists
    df[classification_task] = np.empty((df.shape[0], 0)).tolist()

    # For each application type in list, add label to nn_applications field
    for application_type in topic_list:
        df[classification_task] = df.apply(
            lambda row: row[classification_task] + [application_type] if row[application_type] == 1 else row[
                classification_task], axis=1)

    if drop_columns:
        df.drop(topic_list, axis=1, inplace=True)

    return df


def get_nn_type_from_architecture(data_frame, type_list=None):
    """
    Analyzes layer information to extract information about network type.

    :return: NN Type labels
    """

    # JOB: Filter by repositories with architecture information
    data_frame = data_frame[(data_frame['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))) | (
        data_frame['py_data'].apply(func=lambda x: x.get('model_file_found')))]

    print('Number of repositories with architecture information: %d' % len(data_frame))

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

    # JOB: Iterate through repositories with architecture information and classify them based on layers
    for ix, (index, repo) in enumerate(data_frame.iterrows()):

        # Initialize label attribution with zeros
        is_conv_nn = 0
        is_recurrent_nn = 0
        is_feed_forward_nn = 0
        layers = []

        # Determine source of architecture information (h5/.py file)
        if repo['h5_data'].get('extracted_architecture'):
            layers = repo['h5_data'].get('model_layers')
        elif repo['py_data'].get('model_file_found'):
            layers = repo['py_data'].get('model_layers')

        # Extract all layer types to set
        layer_types = set([layer.get('layer_type') for layer in layers.values()])

        # Check for intersections with defined layer type sets and determine whether to set label to 1
        if len(layer_types.intersection(convolutional_layers)) > 0:
            is_conv_nn = 1
        if len(layer_types.intersection(recurrent_layers)) > 0:
            is_recurrent_nn = 1

        # If no defined layer type is present, default to feed forward type
        if not (is_recurrent_nn or is_conv_nn):
            is_feed_forward_nn = 1

        # Insert extracted network type information into data frame
        data_frame.loc[[index], ['feed_forward_type', 'conv_type', 'recurrent_type']] = [is_feed_forward_nn, is_conv_nn,
                                                                                         is_recurrent_nn]
        print_progress(ix + 1, len(data_frame))

    return data_frame


def preprocess_data(df, type_list, load_data=False):
    """
    Applies pre-processing and returns train-test split of data set.

    :param type_list: Features to predict
    :param load_data: Flag indicating whether to load pre-trained models
    :param df: Data frame to process
    :return: Train-test split of preprocessed data and fitted count vectorizer
    """

    y = df.loc[:, type_list]

    begin_time = time.time()

    if load_data:
        print('Loading stored matrix and models ...')
        X = sparse.load_npz(os.path.join(ROOT_DIR, 'DataCollection/data/Feature_Matrix.npz'))
        count_vectorizer = joblib.load(os.path.join(ROOT_DIR, 'DataCollection/data/CV_trained.pkl'))
        tfidf_transformer = joblib.load(os.path.join(ROOT_DIR, 'DataCollection/data/tf_idf_transformer.pkl'))

    else:
        # Extract feature matrix
        X = df['readme_text']

        # Fit count vectorizer and transform features

        count_vectorizer = CountVectorizer(strip_accents='unicode', ngram_range=(1, 2),
                                           tokenizer=LemmaTokenizer(), max_features=500).fit(X)
        X = count_vectorizer.transform(X)
        tfidf_transformer = TfidfTransformer(use_idf=True)
        X = tfidf_transformer.fit_transform(X)

        # Save feature matrix
        sparse.save_npz(os.path.join(ROOT_DIR, 'DataCollection/data/Feature_Matrix.npz'), X)

        # Save CountVectorizer
        joblib.dump(count_vectorizer, os.path.join(ROOT_DIR, 'DataCollection/data/CV_trained.pkl'))

        # Save tf-idf transformer
        joblib.dump(tfidf_transformer, os.path.join(ROOT_DIR, 'DataCollection/data/tf_idf_transformer.pkl'))

    end_time = time.time()
    print('Preprocessing/Vectorizing duration: %g seconds' % round((end_time - begin_time), 2))

    print('Shape of data after vectorizing:')
    print(X.shape)
    print(y.shape)

    # Obtain train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    return X_train, X_test, y_train, y_test, X, y, count_vectorizer, tfidf_transformer


def train_model(X_train=None, y_train=None, clf=None, X=None, y=None, cross_validate=False, k=3,load_model=False):
    """
    Trains and returns classifier.

    :param load_model:
    :param clf: Classifier to train
    :param X_train: Features of training set
    :param y_train: Labels of training set
    :return: Trained classifier
    """

    # If model needs to be retrained or trained for the first time
    if not load_model:
        # classifier = OneVsRestClassifier(clf)
        # classifier = BinaryRelevance(clf)
        classifier = LabelPowerset(clf)
        # classifier = ClassifierChain(clf)

        if cross_validate:
            print('Starting Cross Validation using %s ...' % str(classifier))
            predictions = cross_val_predict(classifier, X.astype(float), y.astype(float), cv=k, n_jobs=6,
                                            verbose=2)
            cross_val_accuracy = metrics.f1_score(y, predictions, average='weighted')  # TODO: Change back to 'samples'

            return cross_val_accuracy

        else:
            classifier.fit(X_train.astype(float), y_train.astype(float))
            dump(classifier, os.path.join(ROOT_DIR, 'DataCollection/data/trained_model.joblib'))

    # If trained model can be loaded from file
    else:
        classifier = load(os.path.join(ROOT_DIR, 'DataCollection/data/models/trained_model.joblib'))

    return classifier


def test_model(X_test, y_test, clf):
    """
    Tests model performance on test data and returns test score.

    :param X_test: Test data features
    :param y_test: Test data labels
    :param clf: Trained classifier to test
    :return: Test score
    """
    predictions = clf.predict(X_test.astype(float))

    score = metrics.f1_score(y_test, predictions, average='weighted')

    return score


def apply_model(clf, count_vectorizer, tfidf_transfomer, data_frame, type_names):
    """
    Applies classifier to readme data passed as a pandas series and returns predictions as data frame.

    :param tfidf_transfomer: Pre-trained tf-idf transformer
    :param type_names: List of classes to predict
    :param count_vectorizer: Pre-trained count vectorizer
    :param clf: Trained classifier for network type classification
    :param data_frame: Data frame containing plain text readme strings
    :return: Predictions in data frame
    """

    X = tfidf_transfomer.transform((count_vectorizer.transform(data_frame['readme_text'])))
    predictions = pd.DataFrame(clf.predict(X.astype(float)).todense(), columns=type_names)

    # Set feed_forward type label to 0 if any of the other 2 labels apply
    # predictions.iloc[:, [0]] = predictions.apply(lambda row: 0 if row[1:2].any() else row[[0]],
    #                                              axis=1)

    result_df = pd.concat(
        [data_frame[['_id', 'repo_url', 'readme_text']].reset_index(drop=True), predictions.reset_index(drop=True)],
        ignore_index=True, axis=1)

    result_df.set_axis(labels=['_id', 'URL', 'Readme', *type_names], axis=1, inplace=True)

    return result_df


def train_test_nn_type(data_frame, write_to_db=False):
    """
    Trains multi-label classifier on neural network types.

    :param data_frame: Data source
    :return:
    """

    data_frame.set_index('repo_full_name', inplace=True)
    program_start_time = time.time()
    collection = DataCollection.DataCollection('Repos_New')

    # Specify list of neural network types to consider
    type_list = ['feed_forward_type', 'conv_type', 'recurrent_type']

    # JOB: Extract neural network type information from architecture
    print('Extracting labels from architecture ...')
    df = get_nn_type_from_architecture(data_frame)

    # JOB: Decode extracted architecture
    print('Decoding extracted architecture information ...')
    df = nn_decoding(df, type_list, 'nn_type')

    # JOB: Write extracted application information to database
    if write_to_db:
        print('Writing extracted architecture information into database ...')
        collection.add_many(df['_id'].values, 'nn_type', df['nn_type'].values)

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
    df_learn = df[(df['readme_language'] == 'English') & (df['readme_text'] != None)]

    # Apply text preprocessing and split into training and test data
    print('Preprocessing data ...')
    X_train, X_test, y_train, y_test, X, y, count_vectorizer, tfidf_transformer = preprocess_data(df_learn, type_list,
                                                                                                  load_data=False)

    print('Number of repositories: %d' % df_learn.shape[0])

    # JOB: Train model
    print('Training model ...')
    begin_time = time.time()
    # clf = train_model(X_train, y_train, LinearSVC(max_iter=10000), load_model=False)
    cross_val_f_score = train_model(None, None, LinearSVC(max_iter=10000, class_weight='balanced'), X, y, cross_validate=True, k=10, load_model=False)

    print('Cross-validated score: %g' % round(cross_val_f_score, 4))
    end_time = time.time()
    print('Model fit duration: %g seconds' % round((end_time - begin_time), 2))
    # print('Cross validation accuracy: %g' % round(cross_val_accuracy, 2))

    begin_time = time.time()
    # score = test_model(X_test, y_test, clf) # TODO: Uncomment

    # print('Score: %g' % round(score, 2))

    end_time = time.time()

    print('Model test duration: %g' % round(end_time - begin_time, 2))

    # JOB: Apply model to repositories without application information
    # Filter for test data without labels
    test_data = data_frame[
        (data_frame['readme_language'] == 'English') &
        (data_frame['readme_text'] != '') &
        (data_frame['readme_text'] != None) &
        (~data_frame.index.isin(df_learn.index))]

    # # Filter for test data without labels
    # test_data = data_frame[
    #     (data_frame['readme_language'] == 'English') &
    #     (data_frame['readme_text'] != '') &
    #     (data_frame['readme_text'] != None) &
    #     ~(data_frame['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))) &
    #     ~(data_frame['py_data'].apply(func=lambda x: x.get('model_file_found')))]
    return None  # TODO: Uncomment

    print('Applying model to data ...')
    begin_time = time.time()
    result_df = apply_model(clf, count_vectorizer, tfidf_transformer, test_data, type_list)
    end_time = time.time()
    print('Prediction time: %g seconds' % round((end_time - begin_time), 2))

    print(tabulate(result_df[:10], headers='keys',
                   tablefmt='psql', showindex=True))

    program_end_time = time.time()
    print('Program run time: %g' % round(program_end_time - program_start_time, 2))

    print('Decoding predicted values ...')
    result_df_decoded = nn_decoding(result_df, type_list, 'nn_type')

    print(tabulate(result_df_decoded[:10], headers='keys',
                   tablefmt='psql', showindex=True))

    # JOB: Write predictions into database
    if write_to_db:
        print('Writing predictions into database ...')
        collection.add_many(result_df_decoded['_id'].values, 'suggested_type',
                            result_df_decoded['nn_type'].values)


def train_test_nn_application(data_frame, write_to_db=False):
    """
    Trains and tests multi-label classifier on neural network application.

    :param data_frame: Data frame containing training and test data consisting
    of readme file as plain text and binary labels indicating affiliation to application category.
    :param write_to_db: Flag indicating whether to write network application into database
    :return:
    """

    # print('Performing tf-idf classification ...')
    collection = DataCollection.DataCollection('Repos_New')

    program_start_time = time.time()  # Time training and testing duration

    # Specify list of considered labels for classifier
    topic_list = ['nlp', 'images']

    # JOB: Encode and preprocess data
    # Encode class affiliation for specified labels
    df = nn_application_encoding(data_frame, topic_list)

    df = nn_decoding(df, topic_list, 'nn_applications')

    # JOB: Write extracted application information to database
    if write_to_db:
        collection.add_many(df['_id'].values, 'application', df['nn_applications'].values)

    print('Number of networks with application labels set: %d' % df.shape[0])

    # Filter data by repositories with non-empty English readmes
    df_learn = df[(df['readme_language'] == 'English') & (df['readme_text'] != None)][
        ['readme_text', *topic_list]]

    print('Number of networks with labels and non-empty English readmes: %d' % df_learn.shape[0])

    # Apply text preprocessing and split into training and test data
    print('Preprocessing data ...')
    X_train, X_test, y_train, y_test, X, y, count_vectorizer, tfidf_transfomer = preprocess_data(df_learn, topic_list,
                                                                                                 load_data=False)

    # JOB: Train model
    print('Training model ...')
    begin_time = time.time()
    cross_val_score = train_model(X_train, y_train, clf=LinearSVC(max_iter=10000, class_weight='balanced'), X=X, y=y,
                                  cross_validate=True, k=10, load_model=False)  # TODO: Change back to clf
    end_time = time.time()
    print('Model fit duration: %g seconds' % round((end_time - begin_time), 2))

    print('Cross Validation Score: %g' % round(cross_val_score, 3))

    return None

    # JOB: Test trained model on test data
    score = test_model(X_test, y_test, clf)

    print('Score: %g' % round(score, 2))

    program_end_time = time.time()
    print('Program execution duration: %g' % round(program_end_time - program_start_time, 2))

    # JOB: Apply model to repositories without application information
    # Filter for test data without labels
    test_data = data_frame[
        (data_frame['readme_language'] == 'English') &
        (data_frame['readme_text'] != '') &
        (data_frame['readme_text'] != None) &
        (~data_frame.index.isin(df_learn.index))]

    print('Applying model to test data ...')
    begin_time = time.time()
    result_df = apply_model(clf, count_vectorizer, tfidf_transfomer, test_data, topic_list)
    end_time = time.time()
    print('Prediction time: %g seconds' % round((end_time - begin_time), 2))

    # print(tabulate(result_df.sample(10), headers='keys',
    #                tablefmt='psql', showindex=True))

    print('Decoding predicted values ...')
    result_df_decoded = nn_decoding(result_df, topic_list, 'nn_applications')

    # JOB: Write predictions into database
    if write_to_db:
        print('Writing predictions into database ...')
        collection.add_many(result_df_decoded['_id'].values, 'suggested_application',
                            result_df_decoded['nn_applications'].values)


def train_test_doc2vec_nn_application(data_frame, classification_task, encoding_func):
    """
    Trains and tests doc2vec-based classifier for neural network application.

    :param encoding_func: Encoding function to use for binarizing class information
    :param classification_task: Flag indicating which label the classifier is trained on
    :param data_frame: Data frame containing repositories
    :return: Trained classifier
    """
    print('\nPerforming doc2vec classification ...')
    # Filter repositories for non-empty English readme
    data_frame = data_frame[(data_frame['readme_language'] == 'English') & (data_frame['readme_text'] != None)]
    data_frame.reset_index(inplace=True)

    topic_list = []
    # Encode applications
    if classification_task == 'nn_application':
        topic_list = ['nlp', 'images']
    elif classification_task == 'nn_type':
        topic_list = ['feed_forward_type', 'conv_type', 'recurrent_type']

    df = encoding_func(data_frame, topic_list)
    X_train, X_test, y_train, y_test = train_test_split(df['readme_text'], df[topic_list], test_size=0.2,
                                                        random_state=0, shuffle=True)

    # Train doc2vec model
    print('Performing doc2vec ...')
    model = perform_doc2vec_embedding(data_frame['readme_text'], train_model=True)

    print('Building text corpora ...')
    train_corpus = list(read_corpus(X_train, tokens_only=True))
    test_corpus = list(read_corpus(X_test, tokens_only=True))

    print('Vectorize text using trained embeddings ...')
    X_train = np.array([model.infer_vector(train_corpus[i]) for i in range(len(train_corpus))])
    X_test = np.array([model.infer_vector(test_corpus[i]) for i in range(len(test_corpus))])

    # Training classifier
    print('Training classifier ...')
    begin_time = time.time()
    classifier = train_model(X_train, y_train,
                             LogisticRegression(C=0.1, solver='saga', max_iter=1000, class_weight='balanced'))
    end_time = time.time()
    print('Training duration: %g seconds' % round((end_time - begin_time), 2))

    # JOB: Test trained model on test data
    print('Testing classifier ...')
    score = test_model(X_test, y_test, classifier)

    print('Score: %g' % round(score, 2))

    return model
