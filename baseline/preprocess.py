import numpy as np

def load_dataset(file_dir):

    f_data = open(file_dir, 'r')
    data_raw = f_data.readlines()
    speech_files = dict()

    for each_line in data_raw:
        line_of_split_data = each_line.split()

        # start of a new file: "FAEM0_SI2022  ["
        if len(line_of_split_data) == 2:
            filename = line_of_split_data[0]
            frames=list()
        # audio data
        elif len(line_of_split_data) == 40:
            frames.append(line_of_split_data)
        # last line of audio data
        elif len(line_of_split_data) == 41:
            frames.append(line_of_split_data[:-1])
            speech_files[filename] = np.array(frames, dtype = np.float64)
        # error
        else:
            print('what?:%s'%line_of_split_data)

    return speech_files

def load_labels(file_dir):

    f_data = open(file_dir, 'r')
    data_raw = f_data.readlines()
    labels_files = dict()

    for each_line in data_raw:
        line_of_split_data = each_line.split()
        labels_files[line_of_split_data[0]] = np.array(line_of_split_data[1:], dtype = np.int)

    return labels_files

def sample_data(dataset, labels, n_frames):

    assert len(dataset) == len(labels)
    dataset_idx = np.arange(len(dataset))
    np.random.shuffle(dataset_idx)

    sampled_data = list()
    sampled_labels = list()

    for idx in dataset_idx:
        data = dataset[idx]
        frames_total = data.shape[0]
        if frames_total < n_frames :
            # print(idx)
            continue
        # assert frames_total >= n_frames
        start = np.random.randint(frames_total - n_frames +1)
        end = start + n_frames
        sampled_data.append(data[start:end, :])

        labels = labels[idx]
        sampled_labels.append(labels[start:end])

    sampled_data = np.array(sampled_data)
    sampled_labels = np.array(sampled_labels)

    return sampled_data, sampled_labels

def pick_long_enough_data(dataset, labels, n_frames):

    assert len(dataset) == len(labels)

    long_enough_dataset = dict()
    long_enough_labels = dict()

    for key in dataset.keys():
        frames_total = dataset[key].shape[0]
        if frames_total >= n_frames :
            long_enough_dataset[key] = dataset[key]
            long_enough_labels[key] = labels[key]

    return long_enough_dataset, long_enough_labels

def randomly_concatenate_data(dataset, labels):

    dataset_values = np.array(list(dataset.values()))
    labels_values = np.array(list(labels.values()))

    assert len(dataset_values) == len(labels_values)
    dataset_idx = np.arange(len(dataset_values))
    np.random.shuffle(dataset_idx)

    sampled_data = dataset_values[dataset_idx]
    sampled_labels = labels_values[dataset_idx]

    sampled_data = np.concatenate(sampled_data, axis=0)
    sampled_labels = np.concatenate(sampled_labels, axis=0)

    return sampled_data, sampled_labels

def get_log_p_s(labels, num_of_states = 1920):

    labels = list(labels.values())
    labels = np.concatenate(labels, axis=0)
    num_of_frames = len(labels)
    count_states = np.zeros(num_of_states)

    for state in range(num_of_frames):
        count_states[labels[state]] += 1

    p_s = count_states / num_of_frames

    return np.log(p_s)

def crop_data_into_mini_batches(dataset, labels, mini_batch_size):

    assert dataset.shape[0] == labels.shape[0]
    total_length = dataset.shape[0]

    cropped_dataset = list()
    cropped_labels = list()

    for i in range(total_length // mini_batch_size):

        start = i * mini_batch_size
        end = start + mini_batch_size

        cropped_dataset.append(dataset[start:end])
        cropped_labels.append(labels[start:end])

    cropped_dataset = np.array(cropped_dataset)
    cropped_labels = np.array(cropped_labels)

    return cropped_dataset, cropped_labels

def make_window(dataset, window_size):

    # make windows of 2D input, making a 3D dataset
    padding_length = window_size//2
    num_frames, num_dimension = dataset.shape
    # print('dataset.shape:%s'%str(dataset.shape))
    dataset_ready_to_be_sliced = np.concatenate([ np.zeros((padding_length, num_dimension)), dataset, np.zeros((padding_length, num_dimension)) ], axis = 0)
    windowed_dataset = list()
    # print('dataset_ready_to_be_sliced:%s'%str(dataset_ready_to_be_sliced.shape))

    for i in range(num_frames):
        windowed_dataset.append(dataset_ready_to_be_sliced[i:i+window_size])

    windowed_dataset = np.array(windowed_dataset)
    # print('windowed_dataset:%s'%str(windowed_dataset.shape))

    return windowed_dataset

if __name__ == '__main__':
    data = load_data('data_example.txt')
    print('data_example loaded')
    labels = load_labels('labels_example.txt')
    print('labels_example loaded')
