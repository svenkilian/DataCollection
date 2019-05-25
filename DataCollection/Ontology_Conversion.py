import datetime

from config import ROOT_DIR
import os
from rdflib import Graph, BNode, ConjunctiveGraph, URIRef, Literal, Namespace, RDF, RDFS
from tqdm import tqdm
import pandas as pd


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
    df_github = pd.read_json(os.path.join(ROOT_DIR, 'DataCollection/data/GitHub_Data.json'))

    # Print columns
    print(df_github.columns)

    # Convert license to string
    df_github['licence'] = df_github['licence'].apply(lambda x: str(x))

    # Select only Models without Architectures
    # df_github = df_github[df_github['h5-nr'] == 0]

    # Print repositories with h5 file
    print(df_github[df_github['h5'].notna()][['h5', 'h5-nr']])

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
        # model name example (user+github repo): huohuotm/CarND-BehavioralCloning
        modelName = row['name'][1:].replace('/', '-')
        owner = '/'.join(row['link'].split('/')[:-1])

        # Create Neural Network and assign model name
        nn = URIRef(base + row['name'][1:])
        g.add((nn, RDF.type, ontologyURI.Neural_Network))
        if len(row['description']) > 0:
            g.add((nn, dc.description, Literal(row['description'])))
        g.add((nn, RDFS.label, Literal(modelName)))

        # Assign owner to model
        if owner.startswith('http'):
            g.add((nn, dc.creator, URIRef(owner)))

        # Assign link to model
        if row['link'].startswith('http'):
            g.add((nn, ontologyURI.hasRepositoryLink, URIRef(row['link'])))

        # Assign last modified date
        g.add((nn, dc.modified, Literal(row['update'])))

        # Assign publishing website
        g.add((nn, dc.publisher, URIRef('https://github.com')))
        # stars
        # if isinstance(row['stars'], float) and not math.isnan(float(row['stars'])):
        #    g.add((nn, ontologyURI.stars, Literal(int(row['stars']))))

        # Add license
        if len(row['licence']) > 0 and row['licence'] != 'nan':
            licence = URIRef(ontology + row['licence'].replace('"', ''))
            g.add((nn, dc.license, licence))

        # Add reference links within readme
        if isinstance(row['references'], str) and row['references'] is not None:
            for ref in row['references'].split(';'):
                ref = ref.split(' ')[0].replace('"', '')
                if '<' not in ref and '>' not in ref and ('http:' in ref or 'https:' in ref):
                    g.add((nn, dc.references, URIRef(ref)))

        # Add seeAlso
        if isinstance(row['seeAlso'], str) and row['seeAlso'] is not None:
            for ref in row['seeAlso'].split(';'):
                ref = ref.split(' ')[0].replace('"', '')
                if '<' not in ref and '>' not in ref and ('http:' in ref or 'https:' in ref):
                    g.add((nn, RDFS.seeAlso, URIRef(ref)))

        # TODO Jannik, hier infos über Neural Network Ontologie hinzufügen.
        # Create Input Layer and add name
        # TOOD Jannik: layerName definieren (Bsp.: https://w3id.org/nno/data#gerardnaughton7/EmergingTechProject_dense_2 ; layerName wäre hier dense_2 (dense bzw. layertype kleingeschrieben). Bitte Layer durchnummerieren (dienen der eindeutigen ID); also <<layer>>_<<NR>>)
        # layer = URIRef(base + row['name'][1:] + '_' + layerName)

        # TODO Jannik: layerType definieren. Wäre in unserem Bsp. https://w3id.org/nno/ontology#Dense; also Dense
        # layerType = URIRef(ontology + layerType)
        # g.add((layer, RDF.type, layerType))
        # # layerName ist wie oben (Bspw.: dense_2)
        # g.add((layer, RDFS.label, Literal(layerName)))

        # Connect Layer to Neural Network
        # g.add((nn, ontologyURI.hasLayer, layer))

        # hasLayerSequence (von Layer)
        # TODO Jannik: layerSeq definieren. Ist die Reihenfolge des Layers (also durchnummeriert). Input Layer = 1; zweiter Layer = 2, etc.
        # g.add((layer, ontologyURI.hasLayerSequence, Literal(layerSeq)))

        # Create Neurons
        # TODO Jannik: nrN definieren. Das ist die Anzahl der Neuronen in diesem Layer.
        # g.add((layer, ontologyURI.hasNeuron, Literal(nrN)))

        # Activation function
        # TODO Jannik: actFunc definieren. Das ist die verwendete Aktivierungsfunktion (Bsp.: relu oder merge; alles/immer kleingeschrieben)
        # g.add((layer, ontologyURI.hasActivationFunction, actFunc))

        # Loss Function
        # TODO Jannik modelLoss definieren. Bspw: categorical_crossentropy (alles immer kleingeschrieben)
        # r = URIRef(ontology + modelLoss)
        # g.add((nn, ontologyURI.hasLossFunction, r))

        # hasOptimizer
        # TODO Jannik modelOpt definieren. Bspw: adam (alles immer kleingeschrieben
        # r = URIRef(ontology + modelOpt)
        # g.add((nn, ontologyURI.hasOptimizer, r))

    # Saving to file
    print('Saving file to ontology/{}'.format(output))
    g.serialize(destination='{}.nt'.format(output), format='nt')
    g.serialize(destination='{}.owl'.format(output), format='pretty-xml')
