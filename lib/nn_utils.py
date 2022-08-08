import tensorflow as tf


''' NN utils '''


def make_nn(input_size, output_size, hidden_sizes):
    """ Creates fully connected neural network
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
    """
    layers = [tf.keras.Input(shape=input_size)] + [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32) for h in hidden_sizes]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(layers)


def make_cnn(input_size, output_size, hidden_sizes, kernel_size=3):
    """ Construct neural network consisting of
          one 1d-convolutional layer that utilizes temporal dependences,
          fully connected network

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layer
    """

    cnn_layer = [tf.keras.Input(shape=input_size),tf.keras.layers.Conv1D(hidden_sizes[0], kernel_size=kernel_size,
                                        padding="same", dtype=tf.float32)]
    layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes[1:]]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(cnn_layer + layers)

def make_encoder_cnn(conv_out_channels, kernel_sizes, fc_sizes):
    layers = []
    
    for i in range(len(conv_out_channels)-1):
        layers.append(tf.keras.layers.Conv1D(filters=conv_out_channels[i], kernel_size=kernel_sizes[i], padding="same", dtype=tf.float32))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
    # skip the batch normalization in the last layer
    layers.append(tf.keras.layers.Conv1D(filters=conv_out_channels[-1], kernel_size=kernel_sizes[-1], padding="same", dtype=tf.float32))
    layers.append(tf.keras.layers.LeakyReLU())
    for i in range(len(fc_sizes)):
        layers.append(tf.keras.layers.Dense(units=fc_sizes[i], dtype=tf.float32))
        layers.append(tf.keras.layers.LeakyReLU())
    return tf.keras.Sequential(layers)



def make_decoder_cnn(conv_out_channels, kernel_sizes):
    layers = []
    
    for i in range(len(conv_out_channels)-1):
        layers.append(tf.keras.layers.Conv1D(filters=conv_out_channels[i], kernel_size=kernel_sizes[i], padding="same", dtype=tf.float32))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
    # skip the batch normalization and activation in the last layer
    layers.append(tf.keras.layers.Conv1D(filters=conv_out_channels[-1], kernel_size=kernel_sizes[-1], padding="same", dtype=tf.float32))
    return tf.keras.Sequential(layers)


def deconv_net(deconv_out_channels, deconv_kernel_sizes, strides, output_padding, activations):
    layers = []
    layers.append(tf.keras.layers.Dense(units=256, activation='elu', dtype=tf.float32))
    for i in range(len(deconv_out_channels)):
        layers.append(tf.keras.layers.Conv1DTranspose(filters=deconv_out_channels[i], 
                                                      kernel_size=deconv_kernel_sizes[i], 
                                                      strides=strides[i],
                                                      padding="same", 
                                                      output_padding=output_padding[i],
                                                      activation=activations[i],
                                                      dtype=tf.float32))
    return tf.keras.Sequential(layers)    


def make_2d_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Creates fully convolutional neural network.
        Used as CNN preprocessor for image data (HMNIST, SPRITES)

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layers
    """
    layers = [tf.keras.layers.Conv2D(h, kernel_size=kernel_size, padding="same",
                                     activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes + [output_size]]
    return tf.keras.Sequential(layers)



