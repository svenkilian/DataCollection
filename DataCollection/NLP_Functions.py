import spacy
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(text):
    """
    Cleans text and returns a list of tokens
    :param text: Text to clean and tokenize
    :return: List of tokens
    """
    lda_tokens = []
    parser = English()
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        elif token.orth_.startswith('kera'):
            lda_tokens.append('KERAS')
        else:
            lda_tokens.append(token.lower_)

    return lda_tokens


def get_lemma(word):
    """
    Gets meaning of word if lemma exists
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
    Gets meaning of word if lemma exists
    :param word: Word to lemmatize
    :return: Lemmatized word
    """
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    """
    Prepares text for topic modelling
    :param text: Text to prepare
    :return: Tokenized and preprocessed text
    """
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['license', 'model', 'training', 'network', 'keras', 'KERAS', 'python', 'machine', 'learning',
                       'using', 'neural', 'input', 'train', 'weight', 'example', 'tensorflow', 'docker', 'environment',
                       'layer', 'result', 'validation', 'project', 'create', 'library', 'dataset', 'data', 'val_acc',
                       'val_loss', 'writeup', 'outlier', 'notebook', 'function', 'sample', 'trained', 'neumf',
                       'implementation', 'class', 'weight', 'output', 'download', 'model_data', 'algorithm', 'import',
                       'epoch', 'install', 'script', 'django', 'framwork', 'application', 'client', 'pytorch', 'file',
                       '--model', 'paper', 'feature', 'number', 'python3', 'directory', 'folder', 'based', 'language',
                       'accuracy', 'result', 'layer', 'framework', 'package', 'crispr', 'flask', 'server', 'params',
                       'database', 'y_train', 'default', 'weight'])

    en_stop = set(stop_words)
    tokens = tokenize(text)
    tokens = [str(token) for token in tokens if len(token) > 4]
    tokens = [str(token) for token in tokens if token not in en_stop]
    tokens = [get_lemma2(token) for token in tokens]

    return tokens
