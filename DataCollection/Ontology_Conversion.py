from rdflib import Graph, BNode, ConjunctiveGraph, URIRef, Literal, Namespace, RDF, RDFS
import string
import math
from tqdm import tqdm
import re
import random
import pandas as pd
import urllib
import ssl
import sys
import os
import pickle
import json


class Ontology_Conversion:
    """
    This class implements the conversion of the Repository data base to a RDF graph
    """


if __name__ == '__main__':
    output = 'data'
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

    # # Load pickled data
    # df_github = pd.read_pickle('./data/GithubNew.pkl')
    # df_github['licence'] = df_github['licence'].apply(lambda x: str(x))

    # print(df_github.head())

    with open('C:/Users/Sven/PycharmProjects/DataCollection/DataCollection/data/GithubNew.pkl', 'r') as file:
        data = pickle.load(file)
        json.dump(data, 'GithubNew.json')
