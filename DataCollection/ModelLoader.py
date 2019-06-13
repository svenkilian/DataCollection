"""
This class extracts the model architecture information from a h5 file
"""
import os
import random
import re
import string
import sys
import urllib
from pprint import pprint
from urllib import request
import h5py
import json
import logging
import pandas as pd
import requests
import config
from config import token_counter

logging.getLogger("tensorflow").setLevel(logging.ERROR)
stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
from keras.engine.saving import load_model

sys.stdout = stdout
from HelperFunctions import get_df_from_json, get_access_tokens, check_access_tokens
from config import ROOT_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

neuronStoplist = ['add', 'dot', 'subtract', 'multiply', 'average', 'maximum', 'minimum', 'concatenate']


def get_activation_function(function_name):
    translation_dict = {
        'relu': 'Rectified Linear Unit',
        'linear': 'Linear',
        'elu': 'Exponential Linear Unit',
        'exponential': 'Exponential',
        'selu': 'Scaled Exponential Linear Unit',
        'tanh': 'Hyperbolic Tangent',
        'sigmoid': 'Sigmoid',
        'hard_sigmoid': 'Hard Sigmoid',
        'softmax': 'Softmax',
        'softplus': 'Softplus',
        'softsign': 'Softsign',
    }

    return_name = translation_dict.get(function_name, function_name.capitalize())

    return return_name


def extract_architecture_from_h5(model_url):
    """
    Extracts model architecture from h5 file.

    :param model_url: Link to h5 file on GitHub
    :return:
    """

    sys.stdout.write('\rExtract model architecture from h5 file ...')
    sys.stdout.flush()

    model_url_raw = model_url.replace('blob', 'raw')
    # print(model_url_raw)
    temp_file_name = 'temp_file_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
    temp_file_path = os.path.join(ROOT_DIR, 'DataCollection/data/' + temp_file_name + '.h5')
    path, header = request.urlretrieve(model_url_raw, temp_file_path)

    # print(path)
    sys.stdout.write('\rLoad model from path ...')
    sys.stdout.flush()

    # Set variable to default values
    extracted_architecture = False
    loss_function = None
    optimizer = None
    layers = None

    # print('\n\n\nModel:\n')
    try:
        # Try loading model architecture from file
        model = load_model(path)
    except ValueError as ve:
        print('Value error occurred. %s' % ve.args)
    except SystemError as se:
        print('System error occurred. %s' % se.args)
    except Exception as e:
        print('Unknown exception occurred. %s' % e.args)
    else:
        # In case model could be successfully loaded
        sys.stdout.write('\rLoading model successful ...')
        sys.stdout.flush()

        extracted_architecture = True
        layers = dict()

        # Iterate through layers
        for idx, layer in enumerate(model.layers):
            # print(layer.get_config())
            # print(str(layer))
            layer_name = layer.name
            layer_type = re.search(r'<keras\..+\..+\.(.*?)\s.*', str(layer)).group(1)
            layer_sequence = idx
            activation_function = None
            nr_neurons = None
            try:
                # Extract number of neurons of current layer
                if layer.name not in set(neuronStoplist):
                    nr_neurons = int(layer.input.shape[1])  # Default number of neurons
                    if len(layer.input.shape) == 3:
                        nr_neurons = int(layer.input.shape[1] * layer.input.shape[2])
                    elif len(layer.input.shape) == 4:
                        nr_neurons = int(layer.input.shape[1] * layer.input.shape[2] * layer.input.shape[3])
            except AttributeError as ae:
                nr_neurons = '?'  # Value assigned if extraction fails

            try:
                # Extract activation function
                activation_function = get_activation_function(layer.output._op._op_def.name.lower())
            except Exception as e:
                pass

            # Build layer as dictionary entry
            layers[str(layer_sequence)] = {'layer_name': layer_name, 'layer_type': layer_type,
                                           'nr_neurons': nr_neurons, 'activation_function': activation_function}

        try:
            loss_function = model.loss.lower()
        except AttributeError as ae:
            # print('Has no loss function.')
            pass

        try:
            optimizer = str(model.optimizer).split()[0].replace('<keras.optimizers.', '')
        except AttributeError as ae:
            # print('Has no optimizer function.')
            pass

    finally:
        os.remove(temp_file_path)  # Delete temporary file
        # print(layers)
        return extracted_architecture, loss_function, optimizer, layers


def extract_architecture_from_python(repo_full_name, tokens):
    """
    Extracts architecture from python file.

    :param repo_full_name: Full name of repository ('owner_name'/'repo_name')
    :param tokens: List of tokens for API access
    :return:
    """

    global loss, layer_type, \
        libraries_found, optimizer, contains_nn, repo_libraries, error_count, optimizerdone, model_file_found

    token_index = config.token_counter % config.rotation_cycle
    config.token_counter += 1
    headers = {'Authorization': 'token ' + tokens[token_index]}

    model_file_found = False
    imports_keras = False
    # List of libraries in the repo
    repo_libraries = []

    # NN characteristics
    layers = dict()
    layer_type = []
    layer_neuron_number = []
    layer_activation = []
    loss = None
    loss_function = None
    optimizer = None

    # Helper variables
    error_count = 0
    optimizerdone = False
    emptylines = 0

    search_terms = ['"import+keras"', '"from+keras"', 'keras.models', 'keras.layers', 'keras.utils',
                    'tf.keras.models.Sequential()', ]
    query_search_terms = '+OR+'.join(search_terms)

    # Request results list for all .py files in repository from GitHub API that import or otherwise use Keras
    response = requests.get(
        'https://api.github.com/search/code?limit=100&per_page=100&q=' + query_search_terms + '+'
                                                                                              'in:file+extension:py+repo:'
        + repo_full_name, headers=headers)
    check_access_tokens(token_index, response)
    json_data = json.loads(response.text)

    py_files_list = []

    try:
        # Iterate through results and extract raw URL to file location
        for file in json_data['items']:
            raw_url = file['html_url'].replace('github.com', 'raw.githubusercontent.com').replace('blob/', '')
            py_files_list.append(raw_url)

    except KeyError as ke:
        pass

    # Print number of .py files found to console
    # print('Number of .py files found: %d' % len(py_files_list))
    # print('Raw URLs: %s\n\n' % ', '.join(py_files_list))

    # Iterate through file URLs and load raw file content
    for ctr, raw_file_url in enumerate(py_files_list):
        # print('Processing file %d/%d ...' % (ctr + 1, len(py_files_list)))
        raw_file = requests.get(raw_file_url).text
        check_access_tokens(token_index, response)
        libraries_found = set()
        try:
            # Store imported libraries in array
            for m in re.finditer(r'^(from|import) (\w+)', raw_file, re.MULTILINE):
                libraries_found.add(m.group(2))

        except Exception as e:
            pass

        # print('\tLibraries: %s' % libraries_found)

        # Check if file imports keras and set model_file to that file if true
        if (('keras' in libraries_found) or ('Keras' in libraries_found)) and not model_file_found:
            model_file = raw_file
            # print('Keras war drin!')
            # Search for 'Sequential' [...] '.compile()' to extract the NN architecture
            sandwich_pattern = r'Sequential(|\s)\(\)(.|\n)*\.compile\([^\)]*\)'
            model_code = re.search(sandwich_pattern, model_file)

            # If code constructing sequential model is found
            if model_code:
                # print('Code snippet found!')
                model_code_lines = model_code.group(0).split('\n')  # split the text in rows

                for indx, line in enumerate(model_code_lines):
                    line = line.strip().replace(' ', '')

                    # Search for model.add parts
                    if re.search(r'\.add\(', line):
                        if not model_file_found:  # Flags the first found model
                            model_file_found = True
                            sys.stdout.write('\rConstruction of Keras model found ...')
                            sys.stdout.flush()
                            # print('Code builds model!')

                        if emptylines <= 5:  # Only look at the first model
                            # print(line)
                            l_type = re.search(r"add\(+(\w)*", line).group(0).replace('((', '(')[4:]
                            if not re.search("#", line[0:10]):  # Skip comment lines
                                emptylines = 0
                                # Filter activation function
                                activation = re.search(r"activation=(\"|\')(\w)*", line)

                                # Temporarily delete every string of form (number, number) to prevent errors
                                # during the extraction of the number of neurons
                                temporary = re.sub(r"\(\s?\d\s?,\s?\d\s?\)", "", line)

                                # Extract number of neurons
                                number_of_neurons = re.search(r"\(\d\d?\d?\d?\s?[,|\)]", temporary)

                                if activation:
                                    activation = activation.group(0)[12:]

                                else:
                                    activation = "Linear"  # Default activation function

                                if number_of_neurons:
                                    number_of_neurons = number_of_neurons.group(0)[1:-1].replace(" ", ""). \
                                        replace(",", "")

                                else:
                                    number_of_neurons = 0

                                if l_type == "Flatten":
                                    try:
                                        # Flatten: same number as previous layer
                                        for previous in reversed(layer_neuron_number):
                                            if previous == 0:
                                                pass
                                            else:
                                                number_of_neurons = previous
                                                break
                                    except:
                                        number_of_neurons = 0

                                if l_type == "Dropout":
                                    number_of_neurons = 0

                                if l_type == "Activation":
                                    activationprev = re.search(r"\([\"|\'].*[\"|\']\)", line)
                                    if activationprev:
                                        layer_activation[-1] = activationprev.group(0)[2:-2]

                                    else:
                                        activationprev = re.search(r"Activation\([A-Za-z]*\(", line)
                                        if activationprev:
                                            layer_activation[-1] = activationprev.group(0)[11:-1]

                                        else:
                                            pass
                                else:

                                    layer_type.append(l_type)
                                    layer_neuron_number.append(number_of_neurons)
                                    layer_activation.append(activation)


                        else:
                            pass
                    elif model_file_found and re.search(r"Sequential\(\)", line):
                        break
                    else:
                        if model_file_found:
                            emptylines += 1

                    if re.search(r"\.compile\(", line) and not optimizerdone:

                        for t in range(indx, len(model_code_lines)):
                            compileline = model_code_lines[t].strip().replace(" ", "")

                            # print(compileline)
                            losstest = re.search(r"loss=[\'|\"]\s?[\w|\s]*[\'|\"]", compileline)
                            optimizertest = re.search(r"optimizer=[\'|\"]\s?[\w|\s]*[\'|\"]", compileline)
                            optimizerkeras = re.search(r"optimizer\s?=\s?keras\.optimizers\..*\(", compileline)
                            optimizerimported = re.search(r"optimizer\s?=.*\(", compileline)

                            if losstest:
                                loss = losstest.group(0)[6:-1].strip()

                            if optimizertest:
                                optimizer = optimizertest.group(0)[11:-1].strip()

                            elif optimizerkeras:
                                optimizer = optimizerkeras.group(0).replace(" ", "")[27:-1]

                            elif optimizerimported:
                                optimizer = optimizerimported.group(0).replace(" ", "")[10:-1]


                            # If the user created an individual optimizer it is searched in the whole file
                            else:
                                optsectry = re.search(r"optimizer\s?=\s?[A-Za-z]*", compileline)
                                if optsectry:
                                    searchforopt = optsectry.group(0)[10:].strip()

                                    individualopt = re.search(searchforopt + r"\s?=.*[\(|\n]", model_file)
                                    if individualopt:
                                        optimizertest2 = re.search(r"=.*\(", individualopt.group(0))
                                        if optimizertest2:  # if not the initialization of the otimizer ist too complex for regex
                                            optimizer = optimizertest2.group(0)[1:-1].strip()
                                            optimizer = optimizer.replace("optimizers.", "").replace("keras.", "")

                            checkend = re.search(r"\)", compileline)
                            if checkend:
                                optimizerdone = True
                                break

            if layer_activation:
                model_file_found = True

            # print('Length layer type: %d' % len(layer_type))
            # print('Length layer activation: %d' % len(layer_activation))
            # print('Length layer neurons: %d' % len(layer_neuron_number))

            for counter in range(len(layer_activation)):
                # Create Input Layer and add name
                l_name = layer_type[counter] + '_' + str(counter + 1)

                # layer_type definieren. Wäre in unserem Bsp. https://w3id.org/nno/ontology#Dense; also Dense
                l_type = layer_type[counter]

                # hasLayerSequence (von Layer)
                # layerSeq definieren. Ist die Reihenfolge des Layers (also durchnummeriert). Input Layer = 1; zweiter Layer = 2, etc.
                layer_sequence = counter

                # Create Neurons
                # nrN definieren. Das ist die Anzahl der Neuronen in diesem Layer.
                nr_neurons = int(layer_neuron_number[counter])

                # Activation function
                # activation_function definieren. Das ist die verwendete Aktivierungsfunktion (Bsp.: relu oder merge; alles/immer kleingeschrieben)
                activation_function = get_activation_function(layer_activation[counter].split(',')[0].replace("'", ""))

                layers[str(layer_sequence)] = {'layer_name': l_name, 'layer_type': l_type,
                                               'nr_neurons': nr_neurons, 'activation_function': activation_function}

            # Loss Function
            # loss_function definieren. Bspw: categorical_crossentropy (alles immer kleingeschrieben)
            if loss is not None:
                loss_function = loss.lower()

            # hasOptimizer
            # modelOpt definieren. Bspw: adam (alles immer kleingeschrieben)
            if optimizer is not None:
                optimizer = optimizer.lower()

        repo_libraries.extend(list(libraries_found))

    # Deduplicate repo_libraries
    repo_libraries = list(dict.fromkeys(repo_libraries))

    if 'keras' in repo_libraries:
        imports_keras = True

    return loss_function, optimizer, layers, imports_keras, model_file_found, repo_libraries


if __name__ == '__main__':
    """
    Main method to be run when file in run directly by Python interpreter
    """

    tokens = get_access_tokens()[0]
    token_counter = 0
    config.rotation_cycle = len(tokens)
    # Data file name with repositories in json format
    data_file_name = 'data.json'

    # Output filepath
    output_filepath = os.path.join(ROOT_DIR, 'data/Layers.json')

    # Get data frame from data
    print('\tGenerating data frame from json file ...')
    data = get_df_from_json(data_file_name)

    # Links to h5 files
    model_links = data[data['h5_files_links'].str.len() > 0].iloc[10:14]['h5_files_links']
    print('\tRetrieve model URLs ...')

    repo_names = data.sample(1)['repo_full_name']

    # # print(model_link)
    # for idx, link in enumerate(model_links):
    #     sys.stdout.write('\rAnalyzing link %g/%g ...' % (idx + 1, len(model_links)))
    #     sys.stdout.flush()
    #     extract_architecture_from_h5(link[0])

    for idx, name in enumerate(repo_names):
        print('Analyzing repository %g/%g ...' % (idx + 1, repo_names.shape[0]))
        loss_function, optimizer, layers, imports_keras, model_file_found, repo_libraries = \
            extract_architecture_from_python('jkwong80/keras_cats_vs_dogs', tokens)

        if model_file_found and len(layers) > 0:
            print('Number of layers: %d' % len(layers))
            print('Loss function: %s' % loss_function)
            print('Optimizer: %s' % optimizer)
            print('Layers: %s' % '\n'.join([str(layers.get(k, {})) for k in layers.keys()]))
            print('Imports Keras: %s' % imports_keras)
            print('\n' * 5)
        else:
            print('No model file found!\n\n\n')

    sys.exit(0)
    # Specify h5 file location and load file
    # filename = 'data/modelforsongsclassificationwithoutstem.h5'
    # f = h5py.File(filename, 'r')
    # url = 'https://github.com//anarchos78/iot-blockchain-ml-botnet-experiment/raw/master/project-files/create_prediction_model/models/botnet_classifier.h5'
    # file_name, headers = urllib.request.urlretrieve(url)
    file_name = 'data/simple_CNN.985-0.66.hdf5'
    f = h5py.File(file_name, 'r')

    ## Traverse attributes
    # def print_attrs(name, obj):
    #     if (obj.attrs.get('model_config') == 'backend' or True):
    #         print(obj.attrs.get('model_config'))
    #         for key, val in obj.attrs.items():
    #             # print('%s: %s' % (key, val))
    #             pass

    # f.visititems(print_attrs)

    # Extract model configuration
    model_config = str(f.attrs.get('model_config')).strip("b\\''")

    # Make dict from string
    d = json.loads(model_config)

    print(d)

    # Initialize list of layer items
    layer_items_list = []

    # Traverse layers from configuration dict
    for element in list(d.get('config', {})):  # .get('layers')):
        # for element in list(d.get('config', {})):
        # print(element)
        layer_type = element.get('class_name', {})
        print('Type: %s' % layer_type)  # Print class name (Type)

        layer_activation = element.get('config', {}).get('activation')
        print('Activation Function: %s' % layer_activation)  # Print activation function

        # Extract number of neurons for layer
        n_neurons = None
        if 'batch_input_shape' in element.get('config', {}):
            n_neurons = element.get('config', {}).get('batch_input_shape')[1]
        elif 'units' in element.get('config', {}):
            n_neurons = element.get('config', {}).get('units', {})
        else:
            n_neurons = None

        if n_neurons is None:
            print('Number of Neurons: %s' % n_neurons)
        else:
            print('Number of Neurons: %d' % n_neurons)
        print()

        # Create layer item
        layer_item = {'type': layer_type, 'activation': layer_activation, 'n_neurons': n_neurons}

        # Append layer item to list
        layer_items_list.append(layer_item)

    # Save layers in output file (json)
    with open(output_filepath, 'w') as outfile:
        json.dump(layer_items_list, outfile)
