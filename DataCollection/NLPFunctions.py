"""
This module implements methods from the domain of natural language processing.
"""
import multiprocessing
import os
import pickle
import time

import gensim
import nltk
import pandas as pd
import pycountry as country
import spacy
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
import gensim.utils as utils
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from polyglot.text import Text
from sklearn.externals import joblib
from spacy.lang.en import English
from tqdm import tqdm

from config import ROOT_DIR

CUSTOM_STOP_WORDS = ['license', 'model', 'training', 'network', 'keras', 'KERAS', 'python', 'machine', 'learning',
                     'using', 'neural', 'input', 'train', 'weight', 'example', 'tensorflow', 'docker', 'environment',
                     'layer', 'result', 'validation', 'project', 'create', 'library', 'dataset', 'data', 'val_acc',
                     'val_loss', 'writeup', 'outlier', 'notebook', 'function', 'sample', 'trained', 'neumf',
                     'implementation', 'class', 'weight', 'output', 'download', 'model_data', 'algorithm', 'import',
                     'epoch', 'install', 'script', 'django', 'framework', 'application', 'client', 'pytorch', 'file',
                     '--model', 'paper', 'feature', 'number', 'python3', 'directory', 'folder', 'based', 'language',
                     'accuracy', 'layer', 'framework', 'crispr', 'flask', 'server', 'params',
                     'database', 'y_train', 'default', 'weight', 'method', 'default', '--act', '--algo', 'evaluate',
                     'accuracy', 'label', 'numpy', '--algo', 'package', 'default', 'framework', 'weight', 'method',
                     'weight', 'example', 'prediction', 'layer', 'activation', '--act', 'y_train'
                     ]


class LemmaTokenizer(object):
    """
    Implements a generator for lemmatized tokens out of a text document
    """

    # Define stop words
    stop_words = nltk.corpus.stopwords.words('english')

    # Add additional domain specific stop words
    stop_words.extend(CUSTOM_STOP_WORDS)

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Generator function returning lemmatized tokens for document
        :param doc:
        :return:
        """
        # Tokenize and lemmatize document
        lemmatized_tokens = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        en_stop = set(self.stop_words)  # Set stop words
        # Filter out short words and defined stop words
        lemmatized_tokens = [token for token in lemmatized_tokens if len(token) >= 2]
        lemmatized_tokens = [token for token in lemmatized_tokens if token not in en_stop]

        return lemmatized_tokens


def tokenize(text):
    """
    Cleans text and returns a list of tokens.

    :param text: Text to clean and tokenize
    :return: List of tokens
    """

    tokens_out = []
    if text:
        # Tokenize non-empty strings
        parser = English()
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            else:
                # Convert tokens to lower case
                tokens_out.append(token.lower_)
    else:
        pass

    return tokens_out


def get_lemma(word):
    """
    Gets meaning of word if lemma exists.

    :param word: Word to identify meaning of
    :return: Lemmatized word
    """

    lemma = wn.morphy(str(word))
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    """
    Gets meaning of word if lemma exists.

    :param word: Word to lemmatize
    :return: Lemmatized word
    """
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    """
    Prepares text for topic modelling.

    :param text: Text to prepare
    :return: Tokenized and preprocessed text
    """

    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(CUSTOM_STOP_WORDS)

    en_stop = set(stop_words)
    tokens = tokenize(text)
    tokens = [str(token) for token in tokens if len(token) >= 4]
    tokens = [str(token) for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]

    return tokens


def preprocess_text(document):
    """
    Performs advanced preprocessing on a string and returns lemmatized list of tokens.

    :param document: Document string to be preprocessed
    :return: List of preprocessed tokens
    """

    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(CUSTOM_STOP_WORDS)
    en_stop = set(stop_words)
    tokens = utils.tokenize(document, lowercase=True, deacc=True)
    tokens = [str(token) for token in tokens if (token not in en_stop and not token.startswith('_'))]
    tokens = [get_lemma2(token) for token in tokens]

    return tokens


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


def deprecated_text_preprocessing(text_series):
    """
    Applies text preprocessing to series containing strings.

    :param text_series: Series with strings
    :return: Series with preprocessed data
    """
    stop_words = nltk.corpus.stopwords.words('english')

    # Remove character digits
    str = '`1234567890-=~@#$%^&*()_+[!{;":\'><.,/?"}]'

    result_array = []

    for text in text_series:
        if text:
            for w in text:
                if w in str:
                    text = text.replace(w, '')
        else:
            pass

        # Tokenize text
        tokens = tokenize(text)

        # Filter out short words and stop words
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatize words
        tokens = [get_lemma(token) for token in tokens]

        text = ' '.join(tokens)
        result_array.append(text)

    return pd.Series(result_array)


def perform_lda(data_frame):
    """
    Performs latent dirichlet allocation on text data in given data frame.

    :param data_frame: Data frame containing Series with text data
    :return: Latent topics
    """

    # Specify readme language as categorical feature
    data_frame['readme_language'].astype('category')

    # Print language distribution (top 10 languages) to console
    # print('\n\n')
    # print(data_frame.groupby(['readme_language']).size().reset_index(name='Count').sort_values(
    #     'Count', ascending=False)[:10])

    # Filter by English readmes
    data_en = data_frame[data_frame['readme_language'] == 'English'].sample(5000)

    # JOB: Apply preprocssing for topic modeling
    spacy.load('en')  # Load English language corpus
    # nltk.download('wordnet')  # Download wordnet
    # nltk.download('stopwords')  # Download stopwords

    # # Initialize empty array of text data
    # text_data = []
    #
    # # Fill text data array
    # print('Performing text preprocessing ...\n')
    # for index, row in tqdm(data_en.iterrows(), total=data_en.shape[0]):
    #     tokens = prepare_text_for_lda(row['readme_text'])
    #     text_data.append(tokens)

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
    print('\nTraining LDA model ...\n')
    n_topics = 4  # Number of topics to extract
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=10,
                                                alpha=[0.01] * n_topics)
    lda_model.save(os.path.join(ROOT_DIR, 'DataCollection/data/model5.gensim'))

    topics = lda_model.print_topics(num_words=5)

    return topics


def tfidf_vectorize(data_frame):
    """
    Creates tf-idf represention for readme files stored in DataFrame
    :param data_frame: DataFrame containing the readme texts to transform
    :return: Vectorized readmes as matrix
    """

    count_vectorizer = joblib.load(os.path.join(ROOT_DIR, 'DataCollection/data/CV_trained.pkl'))
    tfidf_transformer = joblib.load(os.path.join(ROOT_DIR, 'DataCollection/data/tf_idf_transformer.pkl'))

    # Preprocess readmes
    print('Pre-processing readmes ...')
    # Extract feature matrix
    vectorized_readmes = data_frame['readme_text']

    # Fit count vectorizer and transform features
    vectorized_readmes = count_vectorizer.transform(vectorized_readmes)
    vectorized_readmes = tfidf_transformer.fit_transform(vectorized_readmes).todense()

    return vectorized_readmes


def read_corpus(readme_text_series, tokens_only=False):
    """
    Generator function for text pre-processing and tokenization.

    :param readme_text_series: Pandas series containing readme strings.
    :param tokens_only: If true, only return tokenized document, otherwise return tagged document for training embedding
    :return:
    """

    for i, document in tqdm(readme_text_series.iteritems(), total=len(readme_text_series)):
        if tokens_only:
            yield preprocess_text(document)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(preprocess_text(document),
                                                       [i])


def perform_doc2vec_embedding(readme_text_series, train_model=False, vector_size=20, epochs=40, dm_concat=0):
    """
    Performs doc2vec embedding on text corpus from data frame.

    :param readme_text_series: Series containing document corpus in readme_text series
    :param train_model: Flag indicating whether to train doc2vec embedding from scratch
    :param vector_size: Size of representation vectors
    :param epochs: Number of training epochs
    :return: Trained model
    """

    file_name = os.path.join(ROOT_DIR, 'DataCollection/data/doc2vec_model')

    if train_model:
        print('Extracting corpus ...')
        train_corpus = list(read_corpus(readme_text_series))
        # test_corpus = list(read_corpus(data_frame, tokens_only=True))

        # Initialize doc2vec model
        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=5, epochs=epochs,
                                              workers=multiprocessing.cpu_count(), dm_concat=dm_concat, window=6)
        # Build vocabulary
        print('Building vocabulary from corpus ...')
        model.build_vocab(train_corpus)

        # Train model
        print('Training model using %d parallel workers ...' % multiprocessing.cpu_count())
        begin_time = time.time()
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        end_time = time.time()
        print('Training duration: %g seconds.' % round((end_time - begin_time), 2))
        print('Saving model to file ...')
        model.save(file_name)
        print('Successfully saved model to file.')
        print('File location: %s' % file_name)
    else:
        model = Doc2Vec.load(file_name)

    return model
