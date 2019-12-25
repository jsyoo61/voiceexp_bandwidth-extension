import tensorflow as tf

def dense_layer(
    inputs,
    units,
    activation = None,
    kernel_initializer = None,
    name = None
    ):

    dense_layer = tf.layers.dense(
        inputs = inputs,
        units = units,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return dense_layer

def rosie(inputs, reuse = False, scope_name = 'rosie'):

    # inputs have shape [batch_size, mini_batch(time), window_size, num_features]
    shape_of_input = tf.shape(inputs)
    window_inputs = tf.reshape(inputs, shape = (-1, shape_of_input[1], inputs.shape[2]*inputs.shape[3]))

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        # Hidden layers
        h1 = dense_layer(inputs = window_inputs, units = 512, activation = tf.nn.relu, name = 'h1_dense')
        h2 = dense_layer(inputs = h1, units = 1024, activation = tf.nn.relu, name = 'h2_dense')
        h3 = dense_layer(inputs = h2, units = 1024, activation = tf.nn.relu, name = 'h3_dense')

        # Output
        o1 = dense_layer(inputs = h3, units = 1920, activation = None, name = 'output_categorical')

    return o1
