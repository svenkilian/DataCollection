from keras.models import load_model
from rdflib import Graph, BNode, ConjunctiveGraph, URIRef, Literal, Namespace, RDF, RDFS
import rdflib
import string
import math
from tqdm import tqdm
import re
import random
import pandas as pd
import urllib
import ssl


from lxml import html
import pandas as pd
import re
import urllib
import time
import ssl
import re
ssl._create_default_https_context = ssl._create_unverified_context

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options

from itertools import islice

# Use Firefox as Browser for Selenium and set headless = True. If headless = False, then the browser will "actually be opened"
options = Options()
options.headless = True
driver = webdriver.Firefox(options=options)

# Read file and iterate over all rows

ssl._create_default_https_context = ssl._create_unverified_context


types = set()
neuronStoplist = ['add', 'dot', 'subtract', 'multiply', 'average', 'maximum', 'minimum', 'concatenate']


def randomString(N):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))



#TODO Jannik: Evtl. output name verändern (falls gewünscht)?. Zeile 140 + 141 aufpassen, wird in dem Unterordner files derzeit abgespeichert.
output = 'dataMixed'
g = Graph()
ontology = 'https://w3id.org/nno/ontology#'
ontologyURI = Namespace(ontology)
base = 'https://w3id.org/nno/data#'
#baseURI = Namespace(base)
dc = Namespace('http://purl.org/dc/terms/')
owl = Namespace('http://www.w3.org/2002/07/owl#')
vs = Namespace('http://www.w3.org/2003/06/sw-vocab-status/ns#')
cc = Namespace('http://creativecommons.org/ns#')



## General Information about Ontology
tmp = URIRef('https://w3id.org/nno/data')
g.add((tmp, dc.publisher, URIRef('http://www.aifb.kit.edu/web/Web_Science')))
g.add((tmp, owl.versionInfo, Literal(1)))
g.add((tmp, dc.creator, URIRef('http://www.aifb.kit.edu/web/Anna_Nguyen')))
g.add((tmp, dc.creator, URIRef('http://www.aifb.kit.edu/web/Tobias_Weller')))
g.add((tmp, dc.creator, URIRef('http://www.aifb.kit.edu/web/York_Sure-Vetter')))
g.add((tmp, dc.title, Literal('FAIRnets Dataset')))
g.add((tmp, dc.description, Literal('This is the FAIRnets dataset. It contains Information about publicly available Neural Networks.')))
#TODO Jannik evtl. Datum dann später anpassen.
g.add((tmp, dc.issued, Literal('2019-05-09')))
g.add((tmp, RDFS.label, Literal('FAIRnets Dataset')))
g.add((tmp, vs.term_status, Literal('stable')))
g.add((tmp, cc.licence, URIRef('https://creativecommons.org/licenses/by-nc-sa/4.0/')))


def tobi(idx, row, modelH5):
    r = True
    #print('Processing', idx, 'from', len(df_github))
    modelName = row['name'][1:].replace('/', '-')
    owner = '/'.join(row['link'].split('/')[:-1])
    #if idx == 50:
    #    break

    # Create Neural Network
    nn = URIRef(base + row['name'][1:])
    g.add((nn, RDF.type, ontologyURI.Neural_Network))
    if len(row['description']) > 0:
        g.add((nn, dc.description, Literal(row['description'])))
    g.add((nn, RDFS.label, Literal(modelName)))

    #https://github.com/tonybeltramelli/pix2code
    if owner.startswith('http'):
        g.add((nn, dc.creator, URIRef(owner)))
    if row['link'].startswith('http'):
        g.add((nn, ontologyURI.hasRepositoryLink, URIRef(row['link'])))
    g.add((nn, dc.modified, Literal(row['update'])))
    g.add((nn, dc.publisher, URIRef('https://github.com')))
    # stars
    #if isinstance(row['stars'], float) and not math.isnan(float(row['stars'])):
    #    g.add((nn, ontologyURI.stars, Literal(int(row['stars']))))

    # Licence
    if row['licence'] != 'nan':
        licence = URIRef(ontology + row['licence'].replace('"', ''))
        g.add((nn, dc.license, licence))

    # References
    if not isinstance(row['references'], float):
        for ref in row['references']:
            ref = ref.split(' ')[0].replace('"', '')
            if '<' not in ref and '>' not in ref and ('http:' in ref or 'https:' in ref):
                g.add((nn, dc.references, URIRef(ref)))

    # seeAlso
    if not isinstance(row['seeAlso'], float):
        for ref in row['seeAlso']:
            ref = ref.split(' ')[0].replace('"', '')
            if '<' not in ref and '>' not in ref and ('http:' in ref or 'https:' in ref):
                g.add((nn, RDFS.seeAlso, URIRef(ref)))

    # Download and store tmp
    try:
        url = modelH5.replace('blob', 'raw')
        urllib.request.urlretrieve(url, 'file_tmp.h5')
        
        model = load_model('file_tmp.h5')
        # Nr. of Layers
        #print('Nr. of layers:', len(model.layers))
        #print('Input Layers')
        lTmp = ""
        if hasattr(model, 'input_layers'):
            for l in model.input_layers:
                # Create Input Layer and add name
                lTmp = l.name
                layer = URIRef(base + row['name'][1:] + '_' + l.name)
                layerType = re.sub('<keras\..+\..+\.', '', str(l))
                layerType = re.sub('\sobject.*>', '', layerType)
                layerType = URIRef(ontology + layerType)
                g.add((layer, RDF.type, layerType))
                g.add((layer, RDFS.label, Literal(l.name)))

                # Connect Layer to Neural Network
                g.add((nn, ontologyURI.hasLayer, layer))

                # hasLayerSequence (von Layer)
                g.add((layer, ontologyURI.hasLayerSequence, Literal(1)))

                # Create Neurons
                if not any(word in l.name for word in neuronStoplist):
                    nrN = l.input.shape[1]
                    if len(l.input.shape) == 3:
                        nrN = l.input.shape[1] * l.input.shape[2]
                    elif len(l.input.shape) == 4:
                        nrN = l.input.shape[1] * l.input.shape[2] * l.input.shape[3]
                    if str(nrN) != '?':
                        g.add((layer, ontologyURI.hasNeuron, Literal(nrN)))
                    '''
                    for neuron in range(0, nrN):
                        n = URIRef(base + l.name + '_Neuron_' + randomString(3))
                        g.add((n, RDF.type, baseURI.Neuron))
                        g.add((layer, baseURI.hasNeuron, n))
                    '''
        if hasattr(model, 'layers'):
            #print('Hidden Layers')
            for k in range(1, len(model.layers)):
                if lTmp != model.output.name:
                    l = model.layers[k]
                    # Create hidden Layer and add name
                    layer = URIRef(base + row['name'][1:] + '_' + l.name)
                    layerType = re.sub('<keras\..+\..+\.', '', str(l))
                    layerType = re.sub('\sobject.*>', '', layerType)
                    layerType = URIRef(ontology + layerType)
                    types.add(layerType)
                    g.add((layer, RDF.type, layerType))
                    g.add((layer, RDFS.label, Literal(l.name)))

                    # Connect Layer to Neural Network
                    g.add((nn, ontologyURI.hasLayer, layer))

                    # hasLayerSequence (von LayOWLer)
                    g.add((layer, ontologyURI.hasLayerSequence, Literal(k+1)))

                    # hasActivation_function
                    r = model.layers[k - 1].output._op._op_def.name.lower()
                    r = URIRef(ontology + r)
                    g.add((layer, ontologyURI.hasActivationFunction, r))
                    '''
                    if model.layers[k - 1].output._op._op_def.name.lower() == 'relu':
                        r = URIRef(base + 'relu')
                        g.add((layer, baseURI.hasActivation_function, r))
                    elif model.layers[k - 1].output._op._op_def.name.lower() == 'tanh':
                        r = URIRef(base + 'tanh')
                        g.add((layer, baseURI.hasActivation_function, r))
                    elif model.layers[k - 1].output._op._op_def.name.lower() == 'sigmoid':
                        r = URIRef(base + 'sigmoid')
                        g.add((layer, baseURI.hasActivation_function, r))
                    elif model.layers[k - 1].output._op._op_def.name.lower() == 'softmax':
                        r = URIRef(base + 'softmax')
                        g.add((layer, baseURI.hasActivation_function, r))
                    else:
                        r = URIRef(base + 'unknown')
                        g.add((layer, baseURI.hasActivation_function, r))
                    '''

                    # Create Neurons
                    try:
                        if not any(word in l.name for word in neuronStoplist):
                            nrN = l.input.shape[1]
                            if len(l.input.shape) == 3:
                                nrN = l.input.shape[1] * l.input.shape[2]
                            elif len(l.input.shape) == 4:
                                nrN = l.input.shape[1] * l.input.shape[2] * l.input.shape[3]
                            if str(nrN) != '?':
                                g.add((layer, ontologyURI.hasNeuron, Literal(nrN)))
                            '''
                            for neuron in range(0, nrN):
                                n = URIRef(base + l.name + '_Neuron_' + randomString(3))
                                g.add((n, RDF.type, baseURI.Neuron))
                                g.add((layer, baseURI.hasNeuron, n))
                            '''
                    except Exception as e:
                        print('Error ', idx)
                        continue

        if hasattr(model, 'output_layers'):
            #print('Output Layer')
            for l in model.output_layers:
                # Create hidden Layer and add name
                layer = URIRef(base + row['name'][1:] + '_' + l.name)
                layerType = re.sub('<keras\..+\..+\.', '', str(l))
                layerType = re.sub('\sobject.*>', '', layerType)
                layerType = URIRef(ontology + layerType)
                types.add(layerType)
                g.add((layer, RDF.type, layerType))
                g.add((layer, RDFS.label, Literal(l.name)))

                # Connect Layer to Neural Network
                g.add((nn, ontologyURI.hasLayer, layer))

                # hasLayerSequence (von Layer)
                g.add((layer, ontologyURI.hasLayerSequence, Literal(len(model.layers)+1)))

                # hasActivation_function
                r = model.layers[-1].output._op._op_def.name.lower()
                r = URIRef(ontology + r)
                g.add((layer, ontologyURI.hasActivationFunction, r))
                '''
                if model.layers[-1].output._op._op_def.name.lower() == 'relu':
                    r = URIRef(base + 'relu')
                    g.add((layer, baseURI.hasActivation_function, r))
                elif model.layers[1].output._op._op_def.name.lower() == 'tanh':
                    r = URIRef(base + 'tanh')
                    g.add((layer, baseURI.hasActivation_function, r))
                elif model.layers[1].output._op._op_def.name.lower() == 'sigmoid':
                    r = URIRef(base + 'sigmoid')
                    g.add((layer, baseURI.hasActivation_function, r))
                elif model.layers[1].output._op._op_def.name.lower() == 'softmax':
                    r = URIRef(base + 'softmax')
                    g.add((layer, baseURI.hasActivation_function, r))
                else:
                    r = URIRef(base + 'unknown')
                    g.add((layer, baseURI.hasActivation_function, r))
                '''

                # Create Neurons for Output
                if not any(word in l.name for word in neuronStoplist):
                    nrN = l.output_shape[1]
                    if len(l.output_shape) == 3:
                        nrN = l.output_shape[1] * l.output_shape[2]
                    elif len(l.output_shape) == 4:
                        nrN = l.output_shape[1] * l.output_shape[2] * l.output_shape[3]
                    if str(nrN) != '?':
                        g.add((layer, ontologyURI.hasNeuron, Literal(nrN)))
                    '''
                    for neuron in range(0, nrN):
                        n = URIRef(base + l.name + '_Neuron_' + randomString(3))
                        g.add((n, RDF.type, baseURI.Neuron))
                        g.add((layer, baseURI.hasNeuron, n))
                    '''

        # hasLoss_function
        try:
            r = URIRef(ontology + model.loss.lower())
            g.add((nn, ontologyURI.hasLossFunction, r))
        except AttributeError:
            print('Has no Loss function')


        # hasOptimizer
        try:
            opt = str(model.optimizer).split()[0].replace('<keras.optimizers.', '')
            r = URIRef(ontology + opt.lower())
            g.add((nn, ontologyURI.hasOptimizer, r))
        except AttributeError:
            print('Has no Optimizer function')
    except Exception as e:
        print('Could not download h5 file', url)
        r = False
    return r

def jannik(idx, row):
    file = ""
    # Search for py files in the github repository
    driver.get(row['link'] + '/find/master')
    # driver.get('https://github.com/Timthony/self_drive/find/master')
    actions = ActionChains(driver)
    actions.send_keys('.py')
    actions.perform()
    time.sleep(1)
    tree = html.fromstring(driver.page_source)

    # Use xpath to query all results
    searchResults = tree.xpath(
        '//*[@id="tree-browser"]/li/a/@href'
    )

    # Network characteristics
    layer_type = []
    layer_neuronNumber = []
    layer_activation = []
    loss = ""
    optimizer = ""

    # "helper variables"
    error_count = 0
    found_it = False
    optimizerdone = False
    emptylines = 0

    for file in searchResults:

        file = file.replace('https://github', 'https://raw.githubusercontent').replace('/blob/', '/')
        urllib.request.urlretrieve(file, 'file_tmp.py')
        global netdata
        global data
        with open('file_tmp.py', 'r') as myfile:

            #hier gibts sonst einen decoding error
            try:
                data = myfile.read()
            except:
                print("Error Number: {error}".format(error=error_count))
                error_count += 1


            sandwich_pattern = r"Sequential(|\s)\(\)(.|\n)*\.compile\([^/)]*\)"
            netdata = re.search(sandwich_pattern, data)

            if netdata:
                netdata_text=netdata.group(0).split("\n")



                for idx,line in enumerate(netdata_text):
                    line = line.strip().replace(" ","")
                    if re.search(r"\.add\(",line):


                        if found_it==False: #flags the first found model
                            found_it=True

                        if emptylines<=5: #only look at the first model
                            #print(line)
                            type=re.search(r"add\((\w)*",line).group(0)[4:]

                            if not re.search("#",line[0:10]): #skip comment lines
                                emptylines = 0
                                #filter activatin function
                                activation=re.search(r"activation=(\"|\')(\w)*",line)

                                #temporarily delete every string of form (number,number) to prevent errors during the extraction of the number of neurons
                                temporary=re.sub(r"\(\s?\d\s?,\s?\d\s?\)","",line)

                                #extract number of neurons
                                number_of_neurons=re.search(r"\(\d\d?\d?\d?\s?[,|\)]",temporary)

                                if activation:
                                    activation=activation.group(0)[12:]

                                else:
                                    activation="linear" #default activation function



                                if number_of_neurons:
                                    number_of_neurons=number_of_neurons.group(0)[1:-1].replace(" ","").replace(",","")

                                else:
                                    number_of_neurons=0

                                if type == "Flatten":

                                    try:
                                        for previous in reversed(layer_neuronNumber):  # Flatten: same number as previous layer
                                            if previous == 0:
                                                pass
                                            else:
                                                number_of_neurons = previous
                                                break
                                    except:
                                        number_of_neurons = 0

                                if type == "Dropout":
                                    number_of_neurons=0

                                if type == "Activation":
                                    activationprev=re.search(r"\([\"|\'].*[\"|\']\)",line)
                                    if activationprev:
                                        layer_activation[-1]=activationprev.group(0)[2:-2]

                                    else:
                                        activationprev=re.search(r"Activation\([A-Za-z]*\(",line)
                                        if activationprev:
                                            layer_activation[-1]=activationprev.group(0)[11:-1]

                                        else:
                                            pass
                                else:

                                    layer_type.append(type)
                                    layer_neuronNumber.append(number_of_neurons)
                                    layer_activation.append(activation)

                        else:
                            pass

                    else:
                        if found_it:
                            emptylines+=1

                    if re.search(r"\.compile\(",line) and not optimizerdone:


                        for t in range(idx,len(netdata_text)):
                            compileline = netdata_text[t].strip().replace(" ","")



                            #print(compileline)
                            losstest=re.search(r"loss=[\'|\"]\s?[\w|\s]*[\'|\"]",compileline)
                            optimizertest = re.search(r"optimizer=[\'|\"]\s?[\w|\s]*[\'|\"]", compileline)
                            optimizerkeras= re.search(r"optimizer\s?=\s?keras\.optimizers\..*\(",compileline)
                            optimizerimported=re.search(r"optimizer\s?=.*\(",compileline)

                            if losstest:
                                loss=losstest.group(0)[6:-1].strip()

                            if optimizertest:

                                optimizer=optimizertest.group(0)[11:-1].strip()

                            elif optimizerkeras:
                                optimizer=optimizerkeras.group(0).replace(" ","")[27:-1]

                            elif optimizerimported:
                                optimizer=optimizerimported.group(0).replace(" ","")[10:-1]


                            # if the user created an individual optimizer it is searched in the whole file
                            else:

                                optsectry=re.search(r"optimizer\s?=\s?[A-Za-z]*",compileline)
                                if optsectry:
                                    searchforopt=optsectry.group(0)[10:].strip()

                                    individualopt=re.search(searchforopt+r"\s?=.*[\(|\n]",data)
                                    if individualopt:
                                        optimizertest2=re.search(r"=.*\(",individualopt.group(0))
                                        if optimizertest2: # if not the initialization of the otimizer ist too complex for regex
                                            optimizer=optimizertest2.group(0)[1:-1].strip()
                                            optimizer=optimizer.replace("optimizers.","").replace("keras.","")


                            checkend=re.search(r"\)",compileline)
                            if checkend:
                                optimizerdone=True
                                break





    # modelname example (user+github repo): huohuotm/CarND-BehavioralCloning
    modelName = row['name'][1:].replace('/', '-')
    owner = '/'.join(row['link'].split('/')[:-1])


    # Create Neural Network
    nn = URIRef(base + row['name'][1:])
    g.add((nn, RDF.type, ontologyURI.Neural_Network))
    if len(row['description']) > 0:
        g.add((nn, dc.description, Literal(row['description'])))
    g.add((nn, RDFS.label, Literal(modelName)))

    if owner.startswith('http'):
        g.add((nn, dc.creator, URIRef(owner)))
    if row['link'].startswith('http'):
        g.add((nn, ontologyURI.hasRepositoryLink, URIRef(row['link'])))
    g.add((nn, dc.modified, Literal(row['update'])))
    g.add((nn, dc.publisher, URIRef('https://github.com')))
    # stars
    #if isinstance(row['stars'], float) and not math.isnan(float(row['stars'])):
    #    g.add((nn, ontologyURI.stars, Literal(int(row['stars']))))

    # Licence
    if row['licence'] != 'nan':
        licence = URIRef(ontology + row['licence'].replace('"', ''))
        g.add((nn, dc.license, licence))

    # References
    if not isinstance(row['references'], float):
        for ref in row['references']:
            ref = ref.split(' ')[0].replace('"', '')
            if '<' not in ref and '>' not in ref and ('http:' in ref or 'https:' in ref):
                g.add((nn, dc.references, URIRef(ref)))

    # seeAlso
    if not isinstance(row['seeAlso'], float):
        for ref in row['seeAlso']:
            ref = ref.split(' ')[0].replace('"', '')
            if '<' not in ref and '>' not in ref and ('http:' in ref or 'https:' in ref):
                g.add((nn, RDFS.seeAlso, URIRef(ref)))

    #TODO Jannik, hier infos über Neural Network Ontologie hinzufügen.
    #if len(file) > 0:
    #    print(file)
    for banane in range(0,len(layer_activation)):
        print(layer_type[banane])

        # Create Input Layer and add name
        #TOOD Jannik: layerName definieren (Bsp.: https://w3id.org/nno/data#gerardnaughton7/EmergingTechProject_dense_2 ; layerName wäre hier dense_2 (dense bzw. layertype kleingeschrieben). Bitte Layer durchnummerieren (dienen der eindeutigen ID); also <<layer>>_<<NR>>)
        layerName=layer_type[banane]
        layer = URIRef(base + row['name'][1:] + '_' + layerName)

        #TODO Jannik: layerType definieren. Wäre in unserem Bsp. https://w3id.org/nno/ontology#Dense; also Dense
        layerType=layer_type[banane]
        layerType = URIRef(ontology + layerType)
        g.add((layer, RDF.type, layerType))
        #layerName ist wie oben (Bspw.: dense_2)
        g.add((layer, RDFS.label, Literal(layerName)))

        # Connect Layer to Neural Network
        g.add((nn, ontologyURI.hasLayer, layer))

        # hasLayerSequence (von Layer)
        #TODO Jannik: layerSeq definieren. Ist die Reihenfolge des Layers (also durchnummeriert). Input Layer = 1; zweiter Layer = 2, etc.
        layerSeq=banane+1
        g.add((layer, ontologyURI.hasLayerSequence, Literal(layerSeq)))

        # Create Neurons
        #TODO Jannik: nrN definieren. Das ist die Anzahl der Neuronen in diesem Layer.
        nrN=layer_neuronNumber[banane]
        g.add((layer, ontologyURI.hasNeuron, Literal(nrN)))

        # Activation function
        #TODO Jannik: actFunc definieren. Das ist die verwendete Aktivierungsfunktion (Bsp.: relu oder merge; alles/immer kleingeschrieben)
        actFunc=layer_activation[banane].split(',')[0].replace("'","")
        actFunc=URIRef(ontology+actFunc)
        g.add((layer, ontologyURI.hasActivationFunction, actFunc))

    # Loss Function
    #TODO Jannik modelLoss definieren. Bspw: categorical_crossentropy (alles immer kleingeschrieben)
    if len(loss) > 0:
        modelLoss=loss.lower()
        r = URIRef(ontology + modelLoss)
        g.add((nn, ontologyURI.hasLossFunction, r))


    # hasOptimizer
    #TODO Jannik modelOpt definieren. Bspw: adam (alles immer kleingeschrieben
    if len(optimizer) > 0:
        modelOpt=optimizer.lower()
        r = URIRef(ontology + modelOpt)
        g.add((nn, ontologyURI.hasOptimizer, r))

    r = False
    if len(layer_activation) > 0:
        r = True
    return r
# read csv.
df_github = pd.read_csv('./GithubNew.csv', converters={"h5": lambda x: x.replace('\'','').strip("[]").split(", "), 'references': lambda x: x.split(";"), 'seeAlso:': lambda x: x.split})
s = len(df_github)
df_models = pd.read_csv('./models.csv')
df_error = pd.read_csv('./errorMixed.csv')
df_github['licence'] = df_github['licence'].fillna('nan')
df_github['description'] = df_github['description'].fillna('')
df_github = df_github[~df_github['link'].isin(df_error['name'])]
graph = rdflib.Graph()
g = graph.parse('{}.nt'.format(output), format="nt")
counterJannik, counterTobi = 0, 0
counterJannik = len(df_error[df_error['wrapper'] == 'jannik'])
counterTobi = len(df_error[df_error['wrapper'] == 'tobi'])
#df_github = pd.read_pickle('./GithubNew.pkl')

for idx, row in df_github.iterrows():
    print('{git}, Extracting Index {index} from {total}, Jannik: {j}, Tobi: {t}'.format(git=row['link'], index=idx,
                                                                total=s, j=counterJannik, t=counterTobi))

    #try:
    if row['link'] in df_models['link'].values:
        
        r = tobi(idx, row, df_models[df_models['link'] == row['link']]['h5'].iloc[0])
        if r:
            df_error = df_error.append({'name':row['link'], 'wrapper':'tobi', 'error':0}, ignore_index=True)
        else:
            df_error = df_error.append({'name':row['link'], 'wrapper':'tobi', 'error':1}, ignore_index=True)
        counterTobi += 1
    else:
        r = jannik(idx, row)
        if r:
            counterJannik += 1
            df_error = df_error.append({'name':row['link'], 'wrapper':'jannik', 'error':0}, ignore_index=True)
        else:
            df_error = df_error.append({'name':row['link'], 'wrapper':'jannik', 'error':1}, ignore_index=True)
    if idx % 10 == 0:
        g.serialize(destination='{}.nt'.format(output), format='nt')
        #g.serialize(destination='{}.owl'.format(output), format='pretty-xml')
        df_error.to_csv('errorMixed.csv', index=False)
    
    #except Exception as e:
    #    print('Error ', idx)
    #    print(e)
    #    df_error = df_error.append({'name':row['link'], 'error':1}, ignore_index=True)
    #    continue


# Saving to file
print('Saving file to files/{}'.format(output))
g.serialize(destination='{}.nt'.format(output), format='nt')
g.serialize(destination='{}.owl'.format(output), format='pretty-xml')
