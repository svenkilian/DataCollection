"""
This class extracts the model architecture information from a h5 file
"""
import os
import sys
import urllib
from urllib import request

import h5py
import json

from keras.engine.saving import load_model

from HelperFunctions import get_df_from_json
from config import ROOT_DIR


def extract_architecture(model_url):
    """
    Extracts model architecture from h5 file.

    :param model_url: Link to h5 file on GitHub
    :return:
    """

    model_url_raw = model_url.replace('blob', 'raw')
    path, = request.urlretrieve(model_url_raw, os.path.join(ROOT_DIR, 'data/temp_file.h5'))
    model = load_model(path)

if __name__ == '__main__':
    data_file_name = 'data.json'
    output_filepath = os.path.join(ROOT_DIR, 'data/Layers.json')

    data = get_df_from_json(data_file_name)
    model_path = data[data['h5_files_links'].str.len() > 0].iloc[0]['h5_files_links'][0]

    print(model_path)
    extract_architecture(model_path)

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
