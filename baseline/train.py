import os
import numpy as np
import time

from preprocess import *
from model import my_model
# from convert import convert

def train(train_data_dir, train_labels_dir, validation_data_dir, validation_labels_dir, model_dir, model_name, log_dir, output_dir, random_seed):

    np.random.seed(random_seed)

    num_epochs = 300
    mini_batch_size = 216
    batch_size = 216
    learning_rate = 0.01
    num_of_states =1920
    window_size = 9

    # load data
    print('loading data...')

    start_time = time.time()

    # load training data and validation data in dictionary form
    training_dataset = load_dataset(train_data_dir)
    training_labels= load_labels(train_labels_dir)

    # training_dataset_dict = load_dataset(train_data_dir)
    # training_labels_dict = load_labels(train_labels_dir)
    # training_dataset = training_dataset_dict.values()
    # training_labels = training_labels_dict.values()

    print('training data loaded.')

    validation_dataset = load_dataset(validation_data_dir)
    validation_labels = load_labels(validation_labels_dir)

    print('validation data loaded')

    # stacked_training_dataset, stacked_training_labels = stack_data(dataset = training_dataset, labels = training_labels)
    # suitable_training_dataset, suitable_training_labels = pick_long_enough_data(dataset = training_dataset, labels = training_labels, n_frames = n_frames)

    # numpy.one_hot, or function call to get log(p(s))
    log_p_s = get_log_p_s(training_labels, num_of_states = num_of_states)

    end_time = time.time()
    time_elapsed = end_time - start_time

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'log_p_s.npz'), log_p_s = log_p_s)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print('data loaded')
    print('Time elapsed for preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 //60), (time_elapsed % 60 // 1)))

    model = my_model(window_size = window_size)

    for epoch in range(num_epochs):
        print('\nEpoch: %d' % epoch)

        start_time_data_processing = time.time()

        concatenated_training_dataset, concatenated_training_labels = randomly_concatenate_data(dataset = training_dataset, labels = training_labels)
        windowed_training_dataset = make_window(dataset = concatenated_training_dataset, window_size = window_size)
        sampled_training_dataset, sampled_training_labels = crop_data_into_mini_batches(dataset = windowed_training_dataset, labels = concatenated_training_labels, mini_batch_size = mini_batch_size)
        # print('dataset shape: %s, labels shape: %s'%(sampled_training_dataset.shape, sampled_training_labels.shape))

        end_time_data_processing = time.time()

        n_samples = sampled_training_dataset.shape[0]

        start_time_epoch = time.time()

        for i in range(n_samples // batch_size):

            num_iterations = n_samples // batch_size * epoch + i

            start = i * batch_size
            end = (i + 1) * batch_size

            loss, accuracy = model.train(input = sampled_training_dataset[start:end], labels = sampled_training_labels[start:end] , learning_rate = learning_rate)

            if i % 5 == 0:
                print('Iteration: %07d, Loss: %.9f, Accuracy: %.9f'%(num_iterations, loss, accuracy))


        model.save(directory = model_dir, filename = model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch //3600, (time_elapsed_epoch % 3600 //60), (time_elapsed_epoch % 60 //1)))


        print('validation check...')
        concatenated_validation_dataset, concatenated_validation_labels = randomly_concatenate_data(dataset = validation_dataset, labels = validation_labels)
        windowed_validation_dataset = make_window(dataset = concatenated_validation_dataset, window_size = window_size)
        # sampled_validation_dataset, sampled_validation_labels = sample_data(dataset = list(validation_dataset.values()), labels = list(validation_labels.values()), n_frames = n_frames)

        # might have to separate into mini batches
        validation_loss = model.validation_check(validation_input = [windowed_validation_dataset], validation_labels = [concatenated_validation_labels])
        print('validation loss for epoch %d: %.10f' % (epoch, validation_loss))

    # convert(model_dir = model_dir, model_name = model_name, data_dir = test_data_dir, output_dir = output_dir)
    test_result = dict()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_data = load_dataset(data_dir)
    for speaker in test_data.keys():

        windowed_test_data = make_window(dataset = test_data[speaker], window_size = window_size)
        categorized_state = model.test(inputs = [windowed_test_data])[0]
        log_p_s_given_x = np.log(categorized_state)
        log_p_x_given_s = log_p_s_given_x - log_p_s # + log_p_x
        test_result[speaker] = log_p_x_given_s

    pickle.dump(test_result, os.path.join(output_dir, 'lld.pk'))


if __name__ == '__main__':

    # train_data_dir = '..\\train.fbk'
    # train_labels_dir = '..\\train.lab'
    # validation_data_dir = '..\\dev.fbk'
    # validation_labels_dir = '..\\dev.lab'
    # model_dir = '.\\model'
    # model_name = 'model.ckpt'
    # log_dir = '.\\log'
    # output_dir_default = '.\\converted'

    train_data_dir = '../timit_data/train.fbk'
    train_labels_dir = '../timit_data/train.lab'
    validation_data_dir = '../timit_data/dev.fbk'
    validation_labels_dir = '../timit_data/dev.lab'
    model_dir = './model'
    model_name = 'model.ckpt'
    log_dir = './log'
    output_dir_default = './converted'
    random_seed = 0

    train(train_data_dir = train_data_dir, train_labels_dir = train_labels_dir, validation_data_dir = validation_data_dir, validation_labels_dir = validation_labels_dir, model_dir = model_dir, model_name = model_name, log_dir = log_dir, output_dir = output_dir_default, random_seed = random_seed)
