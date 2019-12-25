import pickle
import os
import numpy as np

from model import my_model
from preprocess import load_dataset, make_window

def convert(model_dir, model_name, data_dir, output_dir):

    window_size = 3

    model = my_model(mode = 'test', window_size = window_size)

    model.load(directory = model_dir, filename = model_name)
    loaded_params = np.load(os.path.join(model_dir, 'log_p_s.npz'))
    log_p_s = loaded_params['log_p_s']
    print('model loaded.')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_data = load_dataset(data_dir)
    print('test data loaded, start converting...')

    test_result = dict()

    for speaker in test_data.keys():

        print(speaker)
        windowed_test_data = make_window(dataset = test_data[speaker], window_size = window_size)
        categorized_state = model.test(input = [windowed_test_data])[0]
        log_p_s_given_x = np.log(categorized_state)
        log_p_x_given_s = log_p_s_given_x - log_p_s # + log_p_x
        test_result[speaker] = log_p_x_given_s

    print('conversion complete.')

    pickle.dump(test_result, open(os.path.join(output_dir, 'lld.pk'), 'wb'))


if __name__ == '__main__':

    model_dir_default = './model'
    model_name_default = 'model.ckpt'
    data_dir_default = '../timit_data/test.fbk'
    output_dir_default = './converted'

    convert(model_dir = model_dir_default, model_name = model_name_default, data_dir = data_dir_default, output_dir = output_dir_default)
