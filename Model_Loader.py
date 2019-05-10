import h5py
import json

if __name__ == '__main__':
    output_filepath = 'Layers.json'
    # Specify h5 file location and load file
    filename = 'data/modelforsongsclassificationwithoutstem.h5'
    f = h5py.File(filename, 'r')

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

    # Initialize list of layer items
    layer_items_list = []

    # Traverse layers from configuration dict
    for element in list(d.get('config', {}).get('layers')):
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
