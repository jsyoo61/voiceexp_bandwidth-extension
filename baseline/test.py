import tensorflow as tf
import os

from model import my_model
from preprocess import *

class test_model():

    def __init__(self):

        self.inputs = tf.placeholder(tf.float32, shape = [None, 40])
        self.labels = tf.placeholder(tf.int32, shape = [None])

        self.categorized_state = self.daisy(inputs = self.inputs, reuse = False, scope_name = 'daisy')

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.categorized_state, labels = self.labels))

        self.output_softmax = tf.math.softmax(self.categorized_state)



        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def daisy(self, inputs, reuse = False, scope_name = 'daisy'):

        with tf.variable_scope(scope_name) as scope:
            if reuse == True:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

                h1 = tf.layers.dense(inputs = inputs, units = 1920, activation = None, name = 'test_layer')

        return h1


    def get_loss1(self, inputs, labels):

        loss = self.sess.run([self.loss], feed_dict = {self.inputs : inputs, self.labels : labels})

        return loss

    def test(self, inputs):

        categorized_state = self.sess.run([self.categorized_state], feed_dict = {self.inputs : inputs})

        return categorized_state

v_data = load_dataset('..\\dev.fbk')
v_labels = load_labels('..\\dev.lab')

# c_v_data, c_v_labels = randomly_concatenate_data(dataset = v_data, labels = v_labels)

v_data_keys = list(v_data.keys())
v_label_keys = list(v_label.keys())

test_data = np.array(v_data[v_data_keys[0])
test_label = np.array(v_labels[v_data_keys[0])

model = test_model()

validation_loss = model.get_loss1(inputs = [test_data], labels = [test_label])

print('validation loss: '%validation_loss)
