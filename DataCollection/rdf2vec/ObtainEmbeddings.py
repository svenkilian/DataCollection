'''
This module provides methods to obtain RDF graph embeddings.
'''

import random
import sys
import time
import gensim
import os

from gensim.models import KeyedVectors
from tqdm import tqdm

from config import ROOT_DIR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def random_n_walk_uniform(triples, entity, n_walks, path_depth):
    """
    Creates n_walks random walks of depth path_depth starting from the specified entity.

    :param entity: Entity to start walk from
    :param triples: Triple dict
    :param n_walks: Number of random walks
    :param path_depth: Depth of random walk
    :return:
    """

    path = []

    for k in range(n_walks):
        walk = random_walk_uniform(triples, entity, path_depth)
        path.append(walk)

    return path


def add_triple(net, source, target, edge):
    """
    Adds triple to dictionary.

    :param net: Dictionary
    :param source: Subject
    :param target: Object
    :param edge: Predicate
    :return:
    """

    if source in net:
        if target in net[source]:
            net[source][target].add(edge)
        else:
            net[source][target] = {edge}
    else:
        net[source] = {}
        net[source][target] = {edge}


def get_links(net, source):
    """
    Gets all links of source entity

    :param net: Dictionary of triples
    :param source: Source entity
    :return:
    """
    if source not in net:
        return {}

    return net[source]


# Generate paths (entity->relation->entity) by radom walks
def random_walk_uniform(triples, start_node, max_depth=4):
    """
    Creates random walk starting from start_node

    :param triples: Triple dict
    :param start_node: Node to start walk from
    :param max_depth: Maximum depth of graph walk
    :return:
    """

    next_node = start_node
    path = str(start_node) + '->'
    for i in range(max_depth):
        neighs = get_links(triples, next_node)
        # print (neighs)
        if len(neighs) == 0:
            break

        queue = []
        for neigh in neighs:
            for edge in neighs[neigh]:
                queue.append((edge, neigh))
        edge, next_node = random.choice(queue)
        path = path + str(edge) + '->'
        path = path + str(next_node) + '->'
    path = path.split('->')[:-1]
    return path


def preprocess(fname):
    """
    Preprocesses graph by generating triples

    :param fname: File location of graph
    :return:
    """

    # Initialize triples as empty dict
    triples = {}

    train_counter = 0

    print('File location: %s' % fname)

    with open(fname, 'r') as graph_data:
        for counter, line in enumerate(graph_data):
            if len(line) > 1:
                line = line.rstrip('\n .')
                # print('Line: %s' % line)
                words = line.split(" ")
                # print('Words: %s' % words)
                h = words[0]
                r = words[1]
                t = words[2]

                train_counter += 1  # Increase counter

                # Add triple to dict
                add_triple(triples, h, t, r)

                # print('Triple:', train_counter)

    return triples


def create_embedding(graph_file_name):
    """
    Creates Skip-Gram embedding from RDF graph by first parsing the graph to sentences using random graph walks
    and then training a word2vec language model.

    :param graph_file_name: Name of file containing RDF graph
    :return:
    """

    # Set directory for input graph_file_name and file location
    data_dir = os.path.join(ROOT_DIR, 'DataCollection/data')
    file_location = os.path.join(data_dir, graph_file_name)

    # Create triples from file
    triples = preprocess(file_location)

    # get entities as list
    entities = list(triples.keys())

    # for entity in entities:
    #     print(entity)

    vocabulary = entities
    print('\n\nNumber of entities: %d' % len(vocabulary))

    # Specify number of walks and path depth
    walks = 10
    path_depth = 3

    # print('Entity: %s' % entities[2])

    # Create random walks
    # paths = random_n_walk_uniform(triples, entities[2], walks, path_depth)
    # print('Paths originating from entity:')
    # print(paths)

    # for path in paths:
    #     print(path)
    #     print(len(path))
    #     print()

    # sys.exit(0)

    start_time = time.time()

    # Initialize sentences array
    sentences = []
    for word in tqdm(vocabulary, total=len(vocabulary)):
        sentence = random_n_walk_uniform(triples, word, walks, path_depth)
        if sentence not in sentences:
            sentences.extend(sentence)

        # Deduplicate sentences

    print('Number of sentences: %d' % len(sentences))

    for sentence in sentences[:0]:
        print(sentence)

    elapsed_time = time.time() - start_time
    print('Time elapsed to generate features:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # sg Skip-Gram
    start_time = time.time()
    print('\nStarting SG15 ...')
    model = gensim.models.Word2Vec(size=20, workers=6, window=10, sg=1, negative=15, iter=30)
    model.build_vocab(sentences)
    total_examples = model.corpus_count
    print('Vocabulary size: %d' % len(model.wv.vocab))

    print('\nTraining embedding model ...')
    model.train(sentences=sentences, total_examples=total_examples, epochs=30)
    end_time = time.time()
    print('Time elapsed to train embedding:', time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))

    # sg/cbow features iterations window negative hops random walks

    print('\nSaving embedding model ...')
    model.save(os.path.join(data_dir, 'SG_15'))
    vectors = model.wv

    print('\nSaving vectors ...')
    vectors.save_word2vec_format(os.path.join(data_dir, 'SG_15.txt'), binary=False)


def load_model_and_vectors(model_name):
    """
    Loads pre-trained word2vec model and word vectors.

    :param model_name: Name of model to load
    :return: Model, vocabulary
    """
    data_dir = os.path.join(ROOT_DIR, 'DataCollection/data')
    file_location = os.path.join(data_dir, model_name)
    model = gensim.models.Word2Vec.load(file_location)
    vocabulary = KeyedVectors.load_word2vec_format(file_location + '.txt', binary=False)

    return model, vocabulary


if __name__ == '__main__':

    # create_embedding('graph_architecture.nt')

    print('Loading pre-trained model:')
    trained_model, vectors = load_model_and_vectors('SG_15')

    print('Testing most similar: \n')
    print('<https://w3id.org/nno/data#hepip/CarND-Behavioral-Cloning-P3>')
    most_similar_list = trained_model.most_similar(
        positive=['<https://w3id.org/nno/data#hepip/CarND-Behavioral-Cloning-P3>'], topn=10)

    for item in most_similar_list:
        print(item)

    sys.exit(0)

    # fasttext 500
    print("Starting fast 10 ...")
    modelf = gensim.models.FastText(size=10, workers=5, window=10, sg=1, negative=15, iter=30)
    modelf.build_vocab(sentences)
    # print("vocabulary: ",len(model.wv.vocab))
    total_examples = modelf.corpus_count
    modelf.train(sentences=sentences, total_examples=total_examples, epochs=30)
    # sg/cbow features iterations window negative hops random walks
    modelf.save('FT_10')
    vectorsf = modelf.wv
    vectorsf.save_word2vec_format(os.path.join(data_dir, ' FT_10.txt'), binary=False)

''' 
#GloVe
corpus = Corpus()
corpus.fit(sentences, window=10)
corpus.fit(sentences, window=10)
glove_500 = Glove(no_components=10, learning_rate=0.05)
glove_500.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove_500.add_dictionary(corpus.dictionary)
glove_500.save('glove_10.model')

#GloVe 
glove_200 = Glove(no_components=15, learning_rate=0.05)
glove_200.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove_500.add_dictionary(corpus.dictionary)
glove_500.save('glove_15.model')




del modelf
# fasttext 200
print("start fast 15")
modelf2 = gensim.models.FastText(size=15, workers=5, window=10, sg=1, negative=15, iter=30)
modelf2.build_vocab(sentences)
total_examples = modelf2.corpus_count
modelf2.train(sentences=sentences, total_examples=total_examples, epochs=30)
# sg/cbow features iterations window negative hops random walks
modelf2.save('FT_15')
vectorsf2 = modelf2.wv
vectorsf2.save_word2vec_format(owndata + "FT_15.txt", binary=False)

del modelf2

# sg Skip-Gram
print("start sg 15")
model = gensim.models.Word2Vec(size=5, workers=5, window=10, sg=1, negative=15, iter=30)
model.build_vocab(sentences)
total_examples = model.corpus_count
print("vocabulary: ", len(model.wv.vocab))
model.train(sentences=sentences, total_examples=total_examples, epochs=30)
# sg/cbow features iterations window negative hops random walks
model.save('SG_15')
vectors = model.wv
vectors.save_word2vec_format(owndata + "SG_15.txt", binary=False)

# sg Skip-Gram
model1 = gensim.models.Word2Vec(size=3, workers=5, window=5, sg=1, negative=15, iter=30)
model1.reset_from(model)

# sg Skip-Gram
model2 = gensim.models.Word2Vec(size=3, workers=5, window=5, sg=0, iter=30, cbow_mean=1, alpha=0.05)
model2.reset_from(model)

# sg Skip-Gram
model3 = gensim.models.Word2Vec(size=5, workers=5, window=5, sg=0, iter=30, cbow_mean=1, alpha=0.05)
model3.reset_from(model)

del model
print("start sg 10")
model1.train(sentences=sentences, total_examples=total_examples, epochs=30)
model1.save('SG_10')
vectors1 = model1.wv
vectors1.save_word2vec_format(owndata + "SG_10.txt", binary=False)

del model1
print("start cbow 10")
model2.train(sentences=sentences, total_examples=total_examples, epochs=30)
model2.save('CBOW_10')
vectors2 = model2.wv
vectors2.save_word2vec_format(owndata + "CBOW_10.txt", binary=False)

del model2
print("start cbow 15")
model3.train(sentences=sentences, total_examples=total_examples, epochs=30)
model3.save('CBOW_15')
vectors3 = model3.wv
vectors3.save_word2vec_format(owndata + "CBOW_15.txt", binary=False)
'''
