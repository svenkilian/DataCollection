# Module implementing the conversion of a non-relational database with repositories to an RDF graph structure.

import datetime
import sys

from werkzeug.urls import url_fix

from HelperFunctions import get_loss_function_name, get_optimizer_name, get_layer_type_name, load_data_to_df
from config import ROOT_DIR
import os
from rdflib import Graph, BNode, ConjunctiveGraph, URIRef, Literal, Namespace, RDF, RDFS
from tqdm import tqdm
import numpy as np
import pandas as pd
from tabulate import tabulate
import logging
import html5lib

logging.basicConfig(level=logging.INFO)


def create_rdf_from_df(data_frame, output_name, architecture_filter=False):
    """
    Main method to be run on execution of file.

    :param data_frame: Data frame with repository information
    :param output_name: Output file name
    """

    df_github = data_frame
    # JOB: Define Ontology and namespaces
    output = os.path.join(ROOT_DIR, 'DataCollection/data/' + output_name)  # Specify output name
    g = Graph()  # Instantiate graph

    # Parse neural network ontology
    g.parse('http://people.aifb.kit.edu/ns1888/nno/nno.owl', format='xml')

    ontology = 'https://w3id.org/nno/ontology#'  # Specify ontology locations
    base = 'https://w3id.org/nno/data#'  # Specify base

    # Define namespaces
    ontologyURI = Namespace(ontology)  # Create ontology namespace
    baseURI = Namespace(base)
    dc = Namespace('http://purl.org/dc/terms/')
    owl = Namespace('http://www.w3.org/2002/07/owl#')
    vs = Namespace('http://www.w3.org/2003/06/sw-vocab-status/ns#')
    cc = Namespace('http://creativecommons.org/ns#')
    doap = Namespace('http://usefulinc.com/ns/doap#')

    # General Information about Ontology
    tmp = URIRef('https://w3id.org/nno/data')
    g.add((tmp, dc.publisher, URIRef('http://www.aifb.kit.edu/web/Web_Science')))
    g.add((tmp, owl.versionInfo, Literal(1)))
    g.add((tmp, dc.creator, URIRef('http://www.aifb.kit.edu/web/Anna_Nguyen')))
    g.add((tmp, dc.creator, URIRef('http://www.aifb.kit.edu/web/Tobias_Weller')))
    g.add((tmp, dc.creator, URIRef('http://www.aifb.kit.edu/web/York_Sure-Vetter')))
    g.add((tmp, dc.title, Literal('FAIRnets Dataset')))
    g.add((tmp, dc.description,
           Literal('This is the FAIRnets dataset. It contains Information about publicly available Neural Networks.')))

    date_today = datetime.date.today().isoformat()
    g.add((tmp, dc.issued, Literal(date_today)))
    g.add((tmp, RDFS.label, Literal('FAIRnets Dataset')))
    g.add((tmp, vs.term_status, Literal('stable')))
    g.add((tmp, cc.licence, URIRef('https://creativecommons.org/licenses/by-nc-sa/4.0/')))

    # JOB: List all defined activation functions
    activation_functions = g.subjects(RDF.type, URIRef(ontology + 'Activation_Function'))
    activation_functions_set = set()

    for func in activation_functions:
        activation_functions_set.add(str(func))

    for item in activation_functions_set:
        pass
        # print(item)

    # JOB: List all defined optimizers
    optimizers = g.subjects(RDF.type, URIRef(ontology + 'Optimizer'))
    optmizers_set = set()

    for opt in optimizers:
        optmizers_set.add(str(opt))

    print('\nOptimizers: ')
    for item in optmizers_set:
        pass
        print(item)

    # JOB: List all regression loss functions
    loss_funtions_regr = g.subjects(RDF.type, URIRef(ontology + 'Regressive_Loss'))
    loss_funtions_class = g.subjects(RDF.type, URIRef(ontology + 'Classification_Loss'))
    loss_functions_set = set()

    for loss in loss_funtions_regr:
        loss_functions_set.add(str(loss))

    print('\n\n')

    for loss in loss_funtions_class:
        loss_functions_set.add(str(loss))

    print('\nLoss functions: ')
    for item in loss_functions_set:
        pass
        print(item)

    # JOB: Iterate through repositories in data base
    for idx, row in tqdm(df_github.iterrows(), total=df_github.shape[0]):
        # Set model name (owner/repo) and owner
        modelName = row['repo_full_name'].replace('/', '-')
        owner = 'https://github.com/' + row['repo_owner']

        # Create Neural Network and assign model name
        nn = URIRef(base + row['repo_full_name'])
        if row.get('nn_type') is not np.nan:
            if 'feed_forward_type' in row.get('nn_type'):
                g.add((nn, RDF.type, ontologyURI.Feed_Forward_Neural_Network))  # Add model type
                # print('Added Feed_Forward_Neural_Network')
            if 'conv_type' in row.get('nn_type'):
                g.add((nn, RDF.type, ontologyURI.Convolutional_Neural_Network))  # Add model type
                # print('Added Convolutional_Neural_Network')
            if 'recurrent_type' in row.get('nn_type'):
                g.add((nn, RDF.type, ontologyURI.Recurrent_Neural_Network))  # Add model type
                # print('Added Recurrent_Neural_Network')
        else:
            # If architecture information is not available, set to default
            g.add((nn, RDF.type, ontologyURI.Neural_Network))  # Add model type
        # g.add((nn, RDF.type, ontologyURI.Neural_Network))  # Add model type
        g.add((nn, RDFS.label, Literal(modelName)))  # Add model name

        # Add description
        if row['repo_desc']:
            g.add((nn, dc.description, Literal(row['repo_desc'])))  # Add description

        # Add Readme
        if row['readme_text']:
            g.add((nn, ontologyURI.hasReadme, Literal(row['readme_text'])))

        # Assign owner to model
        if owner.startswith('http'):
            g.add((nn, dc.creator, URIRef(owner)))

        # Assign http URL model
        if row['repo_url'].startswith('http'):
            g.add((nn, ontologyURI.hasRepositoryLink, URIRef(row['repo_url'])))

        # Assign last modified date
        g.add((nn, dc.modified, Literal(row['repo_last_mod'])))

        # Assign creation date
        g.add((nn, dc.created, Literal(row['repo_created_at'])))

        # Assign publishing website
        g.add((nn, dc.publisher, URIRef('https://github.com')))

        # TODO: Assign categories
        if row['repo_tags']:
            for category_tag in row['repo_tags']:
                g.add((nn, doap.category, Literal(category_tag)))

        # Assign stars
        if row['repo_watch'] is not None:
            g.add((nn, ontologyURI.stars, Literal(int(row['repo_watch']))))

        # Add license
        if row['license'] is not None:
            repo_license = URIRef(row['license_url'])
            g.add((nn, dc.license, repo_license))

        # Add reference links within readme
        if row['reference_list']:
            for ref in row['reference_list']:
                g.add((nn, dc.references, URIRef(ref)))

        # Add see_also_links
        if row['see_also_links']:
            for ref in row['see_also_links']:
                g.add((nn, RDFS.seeAlso, URIRef(ref)))

        # Determine whether architecture information exists
        has_architecture = (row['h5_data'].get('extracted_architecture')) or (
            row['py_data'].get('model_file_found'))

        # Determine source of architecture information
        if row['h5_data'].get('extracted_architecture'):
            data_source = row['h5_data']
        elif row['py_data'].get('model_file_found'):
            data_source = row['py_data']
        else:
            data_source = None

        # Query all defined named activation functions from ontology schmema

        # In case architecture information exists, add to graph
        if has_architecture and data_source.get('model_layers') is not None:
            for layer_id, layer in enumerate(data_source.get('model_layers').values()):
                # Create Input Layer and add name and type
                layer_type = layer.get('layer_type')
                layer_name = layer.get('layer_name')

                if layer_type == '':
                    layer_type = 'Unknown'

                layer_URI = URIRef(base + row['repo_full_name'] + '_' + layer_name.lower())
                layer_type_URI = URIRef(ontology + layer_type)

                g.add((layer_URI, RDF.type, layer_type_URI))
                g.add((layer_URI, RDFS.label, Literal(layer_name)))

                # Define layer sequence
                g.add((layer_URI, ontologyURI.hasLayerSequence, Literal(layer_id + 1)))

                # Add number of neurons
                n_neurons = layer.get('nr_neurons')
                g.add((layer_URI, ontologyURI.hasNeurons, Literal(n_neurons)))

                # Add activation function
                if layer.get('activation_function') is not None:
                    activation_function_full_name = layer.get('activation_function')
                    activation_function = activation_function_full_name.replace(' ', '_').replace('"', '').lower()
                    if activation_function == 'softmaxifclasscount>2elsesigmoid':  # Handle exception
                        activation_function = 'softmax'
                else:
                    activation_function_full_name = 'Linear'
                    activation_function = activation_function_full_name.lower()

                substitution_dict = {
                    'rectified_linear_unit': 'relu',
                    'exponential_linear_unit': 'elu',
                    'scaled_exponential_linear_unit': 'selu',
                    'hyperbolic_tangent': 'tanh',
                }

                # Substitute activation function names
                if activation_function in substitution_dict.keys():
                    activation_function = substitution_dict.get(activation_function)

                # Assign URI
                activation_function_URI = URIRef(ontology + activation_function)

                if str(activation_function_URI) in activation_functions_set:
                    g.add((layer_URI, ontologyURI.hasActivationFunction, activation_function_URI))
                else:
                    pass
                    # print('%s is not in set.' % activation_function_URI)

                # # Add activation function name
                # g.add((activation_function_URI, RDFS.label, Literal(activation_function_full_name)))

                # Connect layer to NN
                g.add((nn, ontologyURI.hasLayer, layer_URI))

            # Add loss function
            if data_source.get('loss_function'):
                loss_function_full_name = data_source.get('loss_function')
                loss_function = loss_function_full_name.replace(' ', '_').lower()

                # Add mean squared error type
                if loss_function == 'mse':
                    loss_function = 'mean_squared_error'

                loss_function_URI = URIRef(ontology + loss_function)

                # Add loss function name
                if str(loss_function_URI) in loss_functions_set:
                    g.add((nn, ontologyURI.hasLossFunction, loss_function_URI))
                else:
                    pass
                    print('%s is not in set' % loss_function_URI)
                    # g.add((loss_function_URI, RDFS.label, Literal(get_loss_function_name(loss_function_full_name))))

            # Add optimizer:
            if data_source.get('optimizer'):
                optimizer_full_name = data_source.get('optimizer')
                optimizer = optimizer_full_name.replace(' ', '_').lower()
                optimizer_URI = URIRef(ontology + optimizer)

                if str(optimizer_URI) in optmizers_set:
                    g.add((nn, ontologyURI.hasOptimizer, optimizer_URI))
                else:
                    pass
                    print('Optimizer %s is not in set.' % optimizer_URI)
                    # Add optimizer name
                    # g.add((optimizer_URI, RDFS.label, Literal(get_optimizer_name(optimizer_full_name))))

    if architecture_filter:
        g = filter_graph(g)
    # Save to file
    print('Saving file to {}'.format(output))
    g.serialize(destination='{}.nt'.format(output), format='nt')
    g.serialize(destination='{}.owl'.format(output), format='pretty-xml')
    print('Successfully saved files.')


def filter_graph(graph):
    ontology = 'https://w3id.org/nno/ontology#'  # Specify ontology locations
    ontologyURI = Namespace(ontology)  # Create ontology namespace
    dc = Namespace('http://purl.org/dc/terms/')
    owl = Namespace('http://www.w3.org/2002/07/owl#')
    doap = Namespace('http://usefulinc.com/ns/doap#')

    predicate_deletion_set = {
        ontologyURI.hasReadme,
        ontologyURI.hasRepositoryLink,
        dc.description,
        dc.creator,
        dc.modified,
        dc.created,
        dc.publisher,
        doap.category,
        ontologyURI.stars,
        dc.license,
        dc.references,
        RDFS.seeAlso,
        RDFS.label,
        RDFS.comment,
        ontologyURI.hasLayerSequence,
        ontologyURI.hasNeuron
    }

    object_deletion_set = {
        owl.Class
    }

    for predicate in predicate_deletion_set:
        graph.remove((None, predicate, None))

    for obj in object_deletion_set:
        graph.remove((None, None, obj))

    return graph


if __name__ == '__main__':
    """
    Main method to be executed on run.
    """

    # Load data from json file
    # df_github = pd.read_json(os.path.join(ROOT_DIR, 'DataCollection/data/data.json'))
    df_github = load_data_to_df(os.path.join(ROOT_DIR, 'DataCollection/data/data.json'), download_data=False)

    # Print column names
    # print(df_github.columns)

    # JOB: Transform license object to string
    license_base = 'https://choosealicense.com/licenses/'
    df_github['license_url'] = df_github['license'].apply(lambda lic: license_base + lic.get('key') if lic else None)
    df_github['license'] = df_github['license'].apply(lambda lic: lic.get('name') if lic else None)

    # JOB: Fix potentially broken URLs
    df_github['see_also_links'] = df_github['see_also_links'].apply(lambda ref_list: [url_fix(ref_link) for ref_link in
                                                                                      ref_list])
    df_github['reference_list'] = df_github['reference_list'].apply(lambda ref_list: [url_fix(ref_link) for ref_link in
                                                                                      ref_list])

    # print(tabulate(df_github.sample(10), headers='keys', tablefmt='psql', showindex=True))
    # print(tabulate(df_github.loc[[12852]], headers='keys', tablefmt='psql', showindex=True))
    # print(type(df_github.loc[12852, 'reference_list']))

    # full_graph = create_rdf_from_df(df_github, 'graph_data')

    data_frame = df_github[(df_github['readme_language'] == 'English') & (df_github['readme_text'] != None) & (
            (df_github['readme_text'].str.len()) > 5000) & ~(
        df_github['repo_name'].str.contains(
            '([Bb]ehavior|[Bb]ehaviour|[Cc]ar|[Cc]lon|[Dd]riv)'))].head(1000)

    face_recognition_repo = df_github[df_github['repo_full_name'] == 'EvilPort2/Face-Recognition']
    traffic_sign_repo = df_github[df_github['repo_full_name'] == 'patirasam/Deep-Learning-CNN-Traffic-Sign-Classifier']

    data_frame.sample(1000).append(face_recognition_repo).append(traffic_sign_repo)
    data_frame.reset_index(inplace=True, drop=True)  # Reset index

    print(data_frame.index)

    # Export filtered dataframe to json
    output_file = os.path.join(ROOT_DIR, 'DataCollection/data/filtered_data.json')  # Specify output name
    data_frame.to_json(output_file)

    create_rdf_from_df(data_frame, 'graph_data_small_new')

    # JOB: Filter by repositories with architecture information
    # architecture_data = df_github[(df_github['h5_data'].apply(func=lambda x: x.get('extracted_architecture'))) | (
    #     df_github['py_data'].apply(func=lambda x: x.get('model_file_found')))]

    # create_rdf_from_df(architecture_data.sample(2).append(face_recognition_repo), 'graph_architecture', architecture_filter=True)
