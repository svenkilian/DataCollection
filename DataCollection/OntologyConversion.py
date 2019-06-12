# Module implementing the conversion of a non-relational database with repositories to an RDF graph structure.

import datetime
import sys

from werkzeug.urls import url_fix

from config import ROOT_DIR
import os
from rdflib import Graph, BNode, ConjunctiveGraph, URIRef, Literal, Namespace, RDF, RDFS
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.INFO)


def create_rdf_from_df(data_frame, output_name):
    """
    Main method to be run on execution of file.
    """

    df_github = data_frame
    # JOB: Define Ontology and namespaces
    output = os.path.join(ROOT_DIR, 'DataCollection/data/' + output_name)  # Specify output name
    g = Graph()  # Instantiate graph
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

    # JOB: Iterate through repositories in data base
    for idx, row in tqdm(df_github.iterrows(), total=df_github.shape[0]):
        # Set model name (owner/repo) and owner
        modelName = row['repo_full_name'].replace('/', '-')
        owner = 'https://github.com/' + row['repo_owner']

        # Create Neural Network and assign model name
        nn = URIRef(base + row['repo_full_name'])
        g.add((nn, RDF.type, ontologyURI.Neural_Network))  # Add model type
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
        has_architecture = (row['h5_data'].get('extracted_architecture') is not None) or (
                row['py_data'].get('model_file_found') is not None)

        # Determine source of architecture information
        if row['h5_data'].get('extracted_architecture') is not None:
            data_source = row['h5_data']
        elif row['py_data'].get('model_file_found') is not None:
            data_source = row['py_data']
        else:
            data_source = None

        # In case architecture information exists, add to graph
        if has_architecture and data_source.get('model_layers') is not None:
            for layer_id, layer in enumerate(data_source.get('model_layers').values()):
                # Create Input Layer and add name and type
                layer_type = layer.get('layer_type')
                layer_name = layer_type + '_' + str(layer_id + 1)

                layer_URI = URIRef(base + row['repo_full_name'] + '_' + layer_name.lower())
                layer_type_URI = URIRef(ontology + layer_type)

                g.add((layer_URI, RDF.type, layer_type_URI))
                g.add((layer_URI, RDFS.label, Literal(layer_name)))

                # Define layer sequence
                g.add((layer_URI, ontologyURI.hasLayerSequence, Literal(layer_id + 1)))

                # Add number of neurons
                n_neurons = layer.get('nr_neurons')
                g.add((layer_URI, ontologyURI.hasNeuron, Literal(n_neurons)))

                # Add activation function
                if layer.get('activation_function') is not None:
                    activation_function = layer.get('activation_function').replace(' ', '_')
                    activation_function_URI = URIRef(ontology + activation_function)
                    g.add((layer_URI, ontologyURI.hasActivationFunction, activation_function_URI))

                # Connect layer to NN
                g.add((nn, ontologyURI.hasLayer, layer_URI))

            # Add loss function
            if row['h5_data'].get('loss_function') is not None:
                loss_function = row['h5_data'].get('loss_function').replace(' ', '_').lower()
                loss_function_URI = URIRef(ontology + loss_function)
                g.add((nn, ontologyURI.hasLossFunction, loss_function_URI))

            # Add optimizer:
            if row['h5_data'].get('optimizer') is not None:
                optimizer = row['h5_data'].get('optimizer').lower()
                optimizer_URI = URIRef(ontology + optimizer)
                g.add((nn, ontologyURI.hasOptimizer, optimizer_URI))

    # Save to file
    print('Saving file to ontology/{}'.format(output))
    g.serialize(destination='{}.nt'.format(output), format='nt')
    g.serialize(destination='{}.owl'.format(output), format='pretty-xml')
    print('Successfully saved files.')


if __name__ == '__main__':
    """
    Main method to be executed on run.
    """

    # Load data from json file
    df_github = pd.read_json(os.path.join(ROOT_DIR, 'DataCollection/data/data.json'))

    # Print column names
    print(df_github.columns)

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

    create_rdf_from_df(df_github, 'graph_data')
    create_rdf_from_df(df_github.sample(1000), 'graph_data_small')