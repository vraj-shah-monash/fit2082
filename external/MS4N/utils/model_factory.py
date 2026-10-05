import model.MS4N as MS4N

import model.RNN_Class as RnnModel


def Model_factory(config, input_shape):
    model_type = config['Model_Type']


    if model_type == 'MS4N':
        model = MS4N.MS4NClassifier(config, num_classes=config['num_labels'],
                                          d_state= 64,num_layers=1,d_model=64,) 
    elif model_type == 'Rnn':
            model = RnnModel.VanillaRNN(config, rnn_type='RNN', num_classes=config['num_labels'])
    elif model_type == 'LSTM':
            model = RnnModel.VanillaRNN(config, rnn_type='LSTM', num_classes=config['num_labels'])
    elif model_type == 'GRU':
            model = RnnModel.VanillaRNN(config, rnn_type='GRU', num_classes=config['num_labels'])
  
    else:
        raise ValueError(f"Model_Type '{model_type}' not recognized.")
    
    
    return model
