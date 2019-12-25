import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from module import rosie
from loss import sparse_categorical_cross_entropy

class my_model(object):

    def __init__(self, window_size, num_dimension = 40, num_of_states = 1920, mode = 'train', log_dir = './log'):

        self.num_dimension = num_dimension
        self.num_of_states = num_of_states
        self.window_size = window_size
        # self.mini_batch_size = mini_batch_size
        self.mode = mode
        # input shape = [batch, time, window_size, num_dimension]
        self.input_shape = [None, None, self.window_size, self.num_dimension]
        # label = [batch, state_indices]
        self.label_shape = [None, None]
        self.model = rosie

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            # self.summaries = self.summary()
            self.summary()

    def build_model(self):

        # placeholders for input
        self.input = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input')
        self.label = tf.placeholder(tf.int32, shape = self.label_shape, name = 'label')

        # raw output for loss
        self.output = self.model(inputs = self.input, reuse = False, scope_name = 'state_categorizer')

        # categorized state for model output
        self.categorized_state = tf.nn.softmax(self.output)

        # calculating accuracy
        self.predicted_state = tf.argmax(self.categorized_state, axis = -1, output_type = tf.int32)
        self.concatenated_predicted_state = tf.reshape(self.predicted_state, shape = [-1])
        self.concatenated_label = tf.reshape(self.label, shape = [-1])
        self.length = tf.shape(self.concatenated_label)
        self.accuracy = tf.divide( tf.reduce_sum( tf.cast( tf.math.equal(self.concatenated_predicted_state, self.concatenated_label), dtype = tf.int32) ) , self.length[0] )

        # label into one-hot vector, [batch, time, state(depth)]
        # self.label_one_hot = tf.one_hot(indices = self.label, depth = self.num_of_states, axis = -1)

        # loss
        # self.loss = l1_loss(y = self.output, y_hat = self.label_one_hot)
        self.loss = sparse_categorical_cross_entropy(logits = self.output, labels = self.label)

        # trainable variables
        self.variables = tf.trainable_variables()

        # made for validation check
        self.validation_input = tf.placeholder(tf.float32, shape = self.input_shape, name = 'validation_input')
        self.validation_label = tf.placeholder(tf.int32, shape = self.label_shape, name = 'validation_label')
        self.validation_output = self.model(inputs = self.validation_input, reuse = True, scope_name = 'state_categorizer')
        # self.validation_label_one_hot = tf.one_hot(indices = self.validation_label, depth = self.num_of_states, axis = -1)
        # self.validation_loss = l1_loss(y = self.validation_output, y_hat = self.validation_label_one_hot)
        self.validation_loss = sparse_categorical_cross_entropy(logits = self.validation_output, labels = self.validation_label)


    def optimizer_initializer(self):

        self.learning_rate = tf.placeholder(tf.float32, shape = None, name = 'learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.9).minimize(self.loss, var_list = self.variables)

    def train(self, input, labels, learning_rate):

        loss, _, model_loss_summary, accuracy, accuracy_summary = self.sess.run( \
        [self.loss, self.optimizer, self.model_loss_summary, self.accuracy, self.accuracy_summary], \
        feed_dict = {self.input : input, self.label : labels, self.learning_rate : learning_rate} \
        )

        # categorized_state = self.test(input)
        # predicted_state = np.argmax(categorized_state, axis = -1)
        # accuracy = np.sum(predicted_state==labels)/labels.reshape(-1).shape[0]
        #
        # accuracy_summary = self.sess.run([])

        self.writer.add_summary(model_loss_summary, self.train_step)
        self.writer.add_summary(accuracy_summary, self.train_step)

        self.train_step += 1

        return loss, accuracy

    def test(self, input):

        categorized_state = self.sess.run(self.categorized_state, feed_dict = {self.input : input})

        return categorized_state

    def validation_check(self, validation_input, validation_labels):

        validation_loss, validation_loss_summary = self.sess.run([self.validation_loss, self.validation_loss_summary], feed_dict = {self.validation_input : validation_input, self.validation_label : validation_labels})
        # print(validation_loss,validation_loss_summary)
        self.writer.add_summary(validation_loss_summary, self.train_step)

        return validation_loss

    def summary(self):

        with tf.name_scope('summaries'):
            self.model_loss_summary = tf.summary.scalar('model_loss', self.loss)
            self.validation_loss_summary = tf.summary.scalar('validation_loss_summary', self.validation_loss)
            self.accuracy_summary = tf.summary.scalar('accuracy_summary', self.accuracy)
            # summaries = tf.summary.merge([self.model_loss_summary, self.validation_loss_summary])

        # return summaries

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

    def load(self, directory, filename):

        self.saver.restore(self.sess, os.path.join(directory, filename))

if __name__ == '__main__':

    model = my_model()
    print('Graph Compile Succeeded.')
