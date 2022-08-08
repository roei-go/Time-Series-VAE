"""

TensorFlow models for use in this project.

"""

from .utils import *
from .nn_utils import *
from .gp_kernel import *
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from tensorflow_probability import math as tfm

# Encoders

class diagonal_encoder(tf.keras.Model):
    def __init__(self, encoder_net, z_size, retain_enc_state=False, **kwargs):
        """ Encoder with factorized Normal posterior over temporal dimension
            Used by disjoint VAE and HI-VAE with Standard Normal prior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(diagonal_encoder, self).__init__()
        self.z_size = int(z_size)
        self.retain_enc_state = retain_enc_state
        self.net = encoder_net

    def __call__(self, x, states=None):
        if self.retain_enc_state:
            enc_outputs = self.net(x,states=states) 
            mapped = enc_outputs[0]
            final_state = enc_outputs[1]
        else:
            mapped = self.net(x) 
        z_dist = tfd.MultivariateNormalDiag(loc=mapped[..., :self.z_size],
                                            scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))
        if self.retain_enc_state:
            return z_dist, final_state
        else:
            return z_dist 


class JointEncoder(tf.keras.Model):
    def __init__(self, z_size, hidden_sizes=(64, 64), window_size=3, transpose=False, **kwargs):
        """ Encoder with 1d-convolutional network and factorized Normal posterior
            Used by joint VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param transpose: True for GP prior | False for Standard Normal prior
        """
        super(JointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = make_cnn(2*z_size, hidden_sizes, window_size)
        self.transpose = transpose

    def __call__(self, x):
        mapped = self.net(x)
        if self.transpose:
            num_dim = len(x.shape.as_list())
            perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
            mapped = tf.transpose(mapped, perm=perm)
            return tfd.MultivariateNormalDiag(
                    loc=mapped[..., :self.z_size, :],
                    scale_diag=tf.nn.softplus(mapped[..., self.z_size:, :]))
        return tfd.MultivariateNormalDiag(
                    loc=mapped[..., :self.z_size],
                    scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))

class gpdir_encoder(tf.keras.Model):
    def __init__(self, encoder_net, gz_dim, dz_dim, seq_len, gp_fc_sizes, dir_fc_sizes, alpha, retain_enc_state=False, **kwargs):
        
        super(gpdir_encoder, self).__init__()
        self.gz_dim = int(gz_dim)
        self.dz_dim = int(dz_dim)
        self.z_dim = self.gz_dim + self.dz_dim
        self.seq_len = int(seq_len)
        self.retain_enc_state = retain_enc_state
        self.alpha = alpha
        self.net = encoder_net

        # define MLP layer for learning the gaussian latent space
        self.gp_fc_layers = []
        if len(gp_fc_sizes) >= 1:
            for i in range(len(gp_fc_sizes)):
                self.gp_fc_layers.append(tf.keras.layers.Dense(units=gp_fc_sizes[i] ,dtype=tf.float32, activation='elu', name=f"gp_mlp_layer_{i}"))
        # add two separate layers for the mean and covariance inference
        self.latent_cov_layer = tf.keras.layers.Dense(units=((self.seq_len * (self.seq_len + 1))//2), activation='elu', dtype=tf.float32, name=f"latent_cov_layer")
        self.latent_mean_layer = tf.keras.layers.Dense(units=self.seq_len ,activation='elu', dtype=tf.float32, name=f"latent_mean_layer")


        # define MLP layer for learning the dirichlet latent space
        self.dir_fc_layers = []
        if len(dir_fc_sizes) >= 1:
            for i in range(len(dir_fc_sizes)):
                self.dir_fc_layers.append(tf.keras.layers.Dense(units=dir_fc_sizes[i] ,dtype=tf.float32, activation='elu', name=f"dir_mlp_layer_{i}"))
        # define a fully connected layer to learn alpha of the dirichlet distribution. Note the activation is a softplus since all alphas should be strictly positive
        self.latent_dirichlet_alpha_layer = tf.keras.layers.Dense(units=self.dz_dim, activation='softmax', dtype=tf.float32, name=f"latent_dirichlet_alpha_layer")

    def call(self, x, states=None):
        # encode the input. The encoded input has shape [batch_size, seq_len, hidden_size]
        if self.retain_enc_state:
             enc_net_outputs = self.net(x,states=states) 
             enc_output = enc_net_outputs[0]
             enc_state = enc_net_outputs[1]
        else:
            enc_output = self.net(x)
        batch_size = enc_output.shape.as_list()[0]
        seq_len    = enc_output.shape.as_list()[1]
        
        # gaussian space path
        xg = enc_output
        # pass through the MLP. the output should be with shape [batch_size, seq_len, gz_dim*m] 
        for i in range(len(self.gp_fc_layers)):
            xg = self.gp_fc_layers[i](xg)

        # reshape the output to shape [batch_size, seq_len, gz_dim, m] 
        xg = tf.reshape(tensor=xg, shape=[batch_size,seq_len,self.gz_dim,-1])
        
        # transpose that tensor to shape [batch_size, gz_dim, seq_len, m]
        xg = tf.transpose(xg, perm=[0,2,1,3]) 
        
        # and reshape again to shape [batch_size, gz_dim, seq_len*m]
        xg = tf.reshape(tensor=xg, shape=[batch_size,self.gz_dim,-1])
        
        # We now have a tensor where for each gaussian z dimension we have a vector with m*seq_len elements. 
        
        # Apply fully connected layers on the gaussian part to obtain mean vector and covariance matrix (full - not diagonal)
        mean = self.latent_mean_layer(xg)
        
        cov_lower_values = self.latent_cov_layer(xg)

        # obtain a lower triangular matrix from the covariance vector we got
        cov_triangular_lower = tfm.fill_triangular(cov_lower_values, upper=False)

        # the gaussian latent space has gz_dim distributions - i.e. normal vectors with latent_space_time_length coordinates
        gz_dist = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov_triangular_lower)

        # dirichlet space path
        xd = enc_output
        # flatten the encoder output. we concatenate the info from all time steps and use it to learn the time series "state" underlying the current window
        # that state is modeled with a vector of length dz_dim
        xd = tf.reshape(xd, shape=[xd.shape[0],-1])
        # pass through the MLP. the output should be with shape [batch_size, output_size] 
        for i in range(len(self.dir_fc_layers)):
            xd = self.dir_fc_layers[i](xd)
            
        
        # Apply fully connected layer on the dirichlet input to obtain a vector of alphas. output shape [batch_size,dz_dim]
        xd = self.latent_dirichlet_alpha_layer(xd)
        dirichlet_alphas = self.alpha * xd
        # The dirichlet space has a single dirichlet distribution which can be sampled to obtain a dz_dim dimensional vector defined over the (dz_dim-1) simplex
        dz_dist = tfd.Dirichlet(dirichlet_alphas)

        if self.retain_enc_state:
            return dz_dist, gz_dist, enc_state
        else:
            return dz_dist, gz_dist


class full_cov_gp_encoder(tf.keras.Model):
    def __init__(self, encoder_net, z_dim, seq_len, retain_enc_state=False, **kwargs):
        
        super(full_cov_gp_encoder, self).__init__()
        self.z_dim = int(z_dim)
        self.seq_len = int(seq_len)
        self.retain_enc_state = retain_enc_state
        self.net = encoder_net
        self.latent_cov_layer = tf.keras.layers.Dense(units=((self.seq_len * (self.seq_len + 1))//2), activation='elu', dtype=tf.float32, name=f"latent_cov_layer")

        self.latent_mean_layer = tf.keras.layers.Dense(units=self.seq_len ,activation='elu', dtype=tf.float32, name=f"latent_mean_layer")
                

    def call(self, x, states=None):
        # encode the input. The encoded input has shape [batch_size, seq_len, z_dim*m]
        if self.retain_enc_state:
             enc_net_outputs = self.net(x,states=states) 
             enc_output = enc_net_outputs[0]
             enc_state = enc_net_outputs[1]
        else:
            enc_output = self.net(x)
        
        batch_size = enc_output.shape.as_list()[0]
        seq_len    = enc_output.shape.as_list()[1]

        # reshape the output to shape [batch_size, seq_len, z_dim, m] 
        enc_output_reshaped_per_z = tf.reshape(tensor=enc_output, shape=[batch_size,seq_len,self.z_dim,-1])
        
        # transpose that tensor to shape [batch_size, z_dim, seq_len, m]
        enc_output_z_first = tf.transpose(enc_output_reshaped_per_z, perm=[0,2,1,3]) 
        
        # and reshape again to shape [batch_size, z_dim, seq_len*m]
        enc_output_z_first_reshaped = tf.reshape(tensor=enc_output_z_first, shape=[batch_size,self.z_dim,-1])
        
        # we now have a tensor where for each z dimension we have a vector with m*seq_len elements, from which we can learn the mean vector and covariance 
        # matrix
        mean = self.latent_mean_layer(enc_output_z_first_reshaped)
        
        cov_lower_values = self.latent_cov_layer(enc_output_z_first_reshaped)

        # obtain a lower triangular matrix from the covariance vector we got
        cov_triangular_lower = tfm.fill_triangular(cov_lower_values, upper=False)

        # the output is basically z_dim distributions - i.e. normal vectors with latent_space_time_length coordinates
        z_dist = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov_triangular_lower)
        if self.retain_enc_state:
            return z_dist, enc_state
        else:
            return z_dist

    def model(self):
        x = tf.keras.Input(shape=self.input_size, batch_size=self.batch_size)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class BandedJointEncoder(tf.keras.Model):
    def __init__(self, encoder_net, z_size, data_type=None, retain_enc_state=False, **kwargs):
        """ Encoder with 1d-convolutional network and multivariate Normal posterior
            Used by GP-VAE with proposed banded covariance matrix
            :param z_size: latent space dimensionality
            
            
            :param data_type: needed for some data specific modifications, e.g:
                tf.nn.softplus is a more common and correct choice, however
                tf.nn.sigmoid provides more stable performance on Physionet dataset
        """
        super(BandedJointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = encoder_net
        self.data_type = data_type
        self.retain_enc_state = retain_enc_state
        #self.net = make_encoder_cnn(conv_out_channels=[128,128,128], kernel_sizes=[5,5,5], fc_sizes=[128,3*z_size])

    def __call__(self, x, states=None):
        # encode the input. The encoded input has shape [batch_size, time_len, 3 * z_dim]
        if self.retain_enc_state:
            enc_output, enc_states = self.net(x,states=states) 
            mapped = enc_output
        else:
            mapped = self.net(x) 

        batch_size = mapped.shape.as_list()[0]
        time_length = mapped.shape.as_list()[1]

        # transpose the encoded input Z and time dimensions: it has now a shape [batch_size, 3 * z_dim, time_len]
        num_dim = len(mapped.shape.as_list())
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        mapped_transposed = tf.transpose(mapped, perm=perm)
        
        # Obtain mean and precision matrix components
        # each time point in the encoded input has 3 * z_dim features. first z_dim features are used to create the mean of the Z distribution
        # so mapped_mean shape is (batch_size, z_dim, time_length)
        mapped_mean = mapped_transposed[:, :self.z_size]
        # and the rest 2 * z_dim features are used to create the precision matrix, so mapped_covar shape is (batch_size, 2 * z_dim, time_length)
        mapped_covar = mapped_transposed[:, self.z_size:]
        # tf.nn.sigmoid provides more stable performance on Physionet dataset
        if self.data_type == 'physionet':
            mapped_covar = tf.nn.sigmoid(mapped_covar)
        else:
            mapped_covar = tf.nn.softplus(mapped_covar)
        # reshaping the precision features array to shape of [batch_size, z_dim, 2 * time_length]
        # i.e. every z variable has now two numbers for every time point
        mapped_reshaped = tf.reshape(mapped_covar, [batch_size, self.z_size, 2*time_length])
        
        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size*(2*time_length-1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2*time_length-1)), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length-1)]), batch_size*self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1,time_length)]), batch_size*self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            # Obtain covariance matrix from precision one
            mapped_values = tf.reshape(mapped_reshaped[:, :, :-1], [-1])
            prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            prec_sparse = tf.sparse.reorder(prec_sparse)
            prec_tril = tf.compat.v1.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            prec_tril = prec_tril + eye
            cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))

        num_dim = len(cov_tril.shape)
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        cov_tril_lower = tf.transpose(cov_tril, perm=perm)
        
        # the output is basically z_dim distributions - i.e. normal vectors with time_length coordinates
        z_dist = tfd.MultivariateNormalTriL(loc=mapped_mean, scale_tril=cov_tril_lower)
        if self.retain_enc_state:
            return z_dist, enc_states
        else:
            return z_dist

class encoder_rnn(tf.keras.Model):
    def __init__(self, input_size, batch_size, rnn_hidden_sizes, fc_sizes, rnn_cell, stateful, return_state=False):
        super(encoder_rnn, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.rnn_hidden_sizes = rnn_hidden_sizes
        self.fc_sizes = fc_sizes
        self.rnn_cell = rnn_cell
        self.stateful = stateful
        self.return_state = return_state
        self.net = self.build_arch()
    
    def build_arch(self):
        layers = [tf.keras.Input(shape=self.input_size, batch_size=self.batch_size)]
        # stacking RNN layers
        for i in range(len(self.rnn_hidden_sizes)):
            if self.rnn_cell == 'lstm':
                layers.append(tf.keras.layers.LSTM(self.rnn_hidden_sizes[i], return_sequences=True, stateful=self.stateful, name=f"LSTM_{i}", return_state=self.return_state))
            elif self.rnn_cell == 'gru':
                layers.append(tf.keras.layers.GRU(self.rnn_hidden_sizes[i], return_sequences=True, stateful=self.stateful, name=f"GRU_{i}", return_state=self.return_state))
            else:
                assert "rnn cell must be either LSTM or GRU"
        if len(self.fc_sizes) >= 1:
            for i in range(len(self.fc_sizes)):
                layers.append(tf.keras.layers.Dense(units=self.fc_sizes[i], dtype=tf.float32))
                layers.append(tf.keras.layers.LeakyReLU())
        return tf.keras.Sequential(layers)

    def call(self, x, state=None):
        if self.return_state:
            output, state = self.net(x)
            return output, state
        else:
            output = self.net(x)
            return output

class decoder_rnn(encoder_rnn):
    def __init__(self, input_size, batch_size, output_dim, rnn_hidden_sizes, rnn_cell, fc_sizes, stateful,return_state=False):
        self.output_dim = output_dim
        super(decoder_rnn, self).__init__(input_size=input_size, 
                                          batch_size=batch_size, 
                                          rnn_hidden_sizes=rnn_hidden_sizes, 
                                          rnn_cell=rnn_cell, 
                                          fc_sizes=fc_sizes, 
                                          stateful=stateful,return_state=return_state)
        # in the decoder, we always have a fully connected layer with linear activation at the output, in order to support any range of output values
        self.net.add(tf.keras.layers.Dense(units=self.output_dim, dtype=tf.float32))



def build_rnn(input_size, batch_size, rnn_hidden_sizes, fc_sizes, activations, rnn_cell, stateful, return_state=False):
    
    inputs = tf.keras.Input(shape=input_size, batch_size=batch_size)
    #init_states = tf.keras.Input(shape=input_size, batch_size=batch_size)
    rnn_layers = []
    # stacking RNN layers
    for i in range(len(rnn_hidden_sizes)):
        if rnn_cell == 'lstm':
            rnn_layers.append(tf.keras.layers.LSTM(rnn_hidden_sizes[i], return_sequences=True, stateful=stateful, name=f"LSTM_{i}", return_state=return_state))
        elif rnn_cell == 'gru':
            rnn_layers.append(tf.keras.layers.GRU(rnn_hidden_sizes[i], return_sequences=True, stateful=stateful, name=f"GRU_{i}", return_state=return_state))
        else:
            assert "rnn cell must be either LSTM or GRU"
    fc_layers = []
    if len(fc_sizes) >= 1:
        for i in range(len(fc_sizes)):
            fc_layers.append(tf.keras.layers.Dense(units=fc_sizes[i], activation=activations[i] ,dtype=tf.float32))
                
    
    
    x = inputs
    final_states = []
    for i in range(len(rnn_layers)):
        if return_state:
            x, final_state = rnn_layers[i](x)
            final_states.append(final_state)
        else:
            x = rnn_layers[i](x)
        
    for i in range(len(fc_layers)):
        x = fc_layers[i](x)
    if return_state:
        final_states = tf.stack(final_states,axis=2)
        model = tf.keras.Model(inputs=inputs, outputs=[x,final_states])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=x)

    return model



class rnn(tf.keras.Model):
    def __init__(self,input_size, batch_size, rnn_hidden_sizes, fc_sizes, activations, rnn_cell, stateful, return_state=False, add_activation_on_last_fc_layer=True, name='enc'):
        super(rnn, self).__init__()
        self.return_state = return_state
        self.input_size = input_size
        self.batch_size = batch_size
        self.rnn_layers = []
        # stacking RNN layers
        for i in range(len(rnn_hidden_sizes)):
            if rnn_cell == 'lstm':
                self.rnn_layers.append(tf.keras.layers.LSTM(rnn_hidden_sizes[i], return_sequences=True, stateful=stateful, name=f"{name}_LSTM_{i}", return_state=return_state))
            elif rnn_cell == 'gru':
                self.rnn_layers.append(tf.keras.layers.GRU(rnn_hidden_sizes[i], return_sequences=True, stateful=stateful, name=f"{name}_GRU_{i}", return_state=return_state))
            else:
                assert "rnn cell must be either LSTM or GRU"
        self.fc_layers = []
        if len(fc_sizes) >= 1:
            for i in range(len(fc_sizes)):
                self.fc_layers.append(tf.keras.layers.Dense(units=fc_sizes[i] ,dtype=tf.float32, name=f"{name}_dense_{i}"))
                if (i < len(fc_sizes)-1) or add_activation_on_last_fc_layer:
                    self.fc_layers.append(tf.keras.layers.LeakyReLU(name=f"{name}_activation_{i}"))
    
    def call(self,inputs,states=None):
        if states is not None:
            assert len(states) == len(self.rnn_layers), "every rnn layer must have an initial state"
        x = inputs
        final_states = []
        for i in range(len(self.rnn_layers)):
            if self.return_state:
                if states is not None:
                    init_state = tf.stop_gradient(tf.identity(states[i]))
                    x, final_state = self.rnn_layers[i](inputs=x,initial_state=init_state)
                else:
                    x, final_state = self.rnn_layers[i](inputs=x)
                final_states.append(final_state)
            else:
                x = self.rnn_layers[i](x)
            
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        if self.return_state:
            return x,final_states
        else:
            return x

    def model(self):
        x = tf.keras.Input(shape=self.input_size, batch_size=self.batch_size)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
        
class cnn_rnn(tf.keras.Model):
    def __init__(self,input_size, batch_size, rnn_hidden_sizes, conv_out_channels, conv_kernel_sizes, fc_sizes, activations, rnn_cell, stateful, return_state=False, name='enc'):
        super(cnn_rnn, self).__init__()
        self.return_state = return_state
        self.input_size = input_size
        self.batch_size = batch_size
        self.cnn_layers = []
        self.rnn_layers = []
        self.fc_layers  = []
        # stacking CNN layers
        for i in range(len(conv_out_channels)):
            self.cnn_layers.append(tf.keras.layers.Conv1D(filters=conv_out_channels[i], 
                                                 kernel_size=conv_kernel_sizes[i], 
                                                 strides=2, 
                                                 padding="same", 
                                                 activation=activations[i],
                                                 dtype=tf.float32))
            
        # stacking RNN layers
        for i in range(len(rnn_hidden_sizes)):
            if rnn_cell == 'lstm':
                self.rnn_layers.append(tf.keras.layers.LSTM(rnn_hidden_sizes[i], return_sequences=True, stateful=stateful, name=f"{name}_LSTM_{i}", return_state=return_state))
            elif rnn_cell == 'gru':
                self.rnn_layers.append(tf.keras.layers.GRU(rnn_hidden_sizes[i], return_sequences=True, stateful=stateful, name=f"{name}_GRU_{i}", return_state=return_state))
            else:
                assert "rnn cell must be either LSTM or GRU"
        
        if len(fc_sizes) >= 1:
            for i in range(len(fc_sizes)):
                self.fc_layers.append(tf.keras.layers.Dense(units=fc_sizes[i] ,dtype=tf.float32, activation='elu', name=f"{name}_dense_{i}"))
                
    
    def call(self,inputs,states=None):
        if states is not None:
            assert len(states) == len(self.rnn_layers), "every rnn layer must have an initial state"
        x = inputs
        for i in range(len(self.cnn_layers)):
            x = self.cnn_layers[i](x)
        # forward pass through the rnn layers
        final_states = []
        for i in range(len(self.rnn_layers)):
            if self.return_state:
                if states is not None:
                    init_state = tf.stop_gradient(tf.identity(states[i]))
                    x, final_state = self.rnn_layers[i](inputs=x,initial_state=init_state)
                else:
                    x, final_state = self.rnn_layers[i](inputs=x)
                final_states.append(final_state)
            else:
                x = self.rnn_layers[i](x)
            
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        if self.return_state:
            return x,final_states
        else:
            return x

    def model(self):
        x = tf.keras.Input(shape=self.input_size, batch_size=self.batch_size)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
        
        
# Decoders


class rnn_deconv(tf.keras.Model):
    def __init__(self,
                 input_size, 
                 batch_size,
                 rnn_cell,
                 rnn_hidden_sizes, 
                 deconv_out_channels, 
                 deconv_kernel_sizes,
                 deconv_strides,
                 deconv_output_padding, 
                 deconv_activations,
                 out_fc_sizes,
                 out_fc_activations,
                 return_state=False, 
                 name='dec'):
        super(rnn_deconv, self).__init__()
        self.return_state = return_state
        self.input_size = input_size
        self.batch_size = batch_size
        self.deconv_layers = []
        self.rnn_layers = []
        self.out_fc_layers = []
        

        # stacking RNN layers
        for i in range(len(rnn_hidden_sizes)):
            if rnn_cell == 'lstm':
                self.rnn_layers.append(tf.keras.layers.LSTM(rnn_hidden_sizes[i], return_sequences=True, stateful=False, name=f"{name}_LSTM_{i}", return_state=return_state))
            elif rnn_cell == 'gru':
                self.rnn_layers.append(tf.keras.layers.GRU(rnn_hidden_sizes[i], return_sequences=True, stateful=False, name=f"{name}_GRU_{i}", return_state=return_state))
            else:
                assert "rnn cell must be either LSTM or GRU"

        # stacking CNN layers
        for i in range(len(deconv_out_channels)):
            self.deconv_layers.append(tf.keras.layers.Conv1DTranspose(filters=deconv_out_channels[i], 
                                                      kernel_size=deconv_kernel_sizes[i], 
                                                      strides=deconv_strides[i],
                                                      padding="same", 
                                                      output_padding=deconv_output_padding[i],
                                                      activation=deconv_activations[i],
                                                      dtype=tf.float32))


        if len(out_fc_sizes) >= 1:
            for i in range(len(out_fc_sizes)):
                self.out_fc_layers.append(tf.keras.layers.Dense(units=out_fc_sizes[i] ,dtype=tf.float32, activation=out_fc_activations[i], name=f"{name}_dense_{i}"))
            
                        
    
    def call(self,inputs,states=None):
        if states is not None:
            assert len(states) == len(self.rnn_layers), "every rnn layer must have an initial state"
        x = inputs
        
        for i in range(len(self.deconv_layers)):
            x = self.deconv_layers[i](x)

        # forward pass through the rnn layers
        final_states = []
        for i in range(len(self.rnn_layers)):
            if self.return_state:
                if states is not None:
                    init_state = tf.stop_gradient(tf.identity(states[i]))
                    x, final_state = self.rnn_layers[i](inputs=x,initial_state=init_state)
                else:
                    x, final_state = self.rnn_layers[i](inputs=x)
                final_states.append(final_state)
            else:
                x = self.rnn_layers[i](x)

        for i in range(len(self.out_fc_layers)):
            x = self.out_fc_layers[i](x)

        
        
        if self.return_state:
            return x,final_states
        else:
            return x

    def model(self):
        x = tf.keras.Input(shape=self.input_size, batch_size=self.batch_size)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class cnn_decoder(tf.keras.Model):
    def __init__(self, conv_out_channels, kernel_sizes):
        """ 
        Decoder parent class with no specified output distribution
        :param output_size: list of output feature map sizes.
        :param kernel_sizes: list of kernel sizes
        """
        super(cnn_decoder, self).__init__()
        self.net = make_decoder_cnn(conv_out_channels, kernel_sizes)

    def __call__(self, x):
        pass

class gaussian_cnn_decoder(cnn_decoder):
    """ 
    Decoder with Gaussian output distribution.
    """
    def __call__(self, x):
        mean = self.net(x)
        var = tf.ones(tf.shape(mean), dtype=tf.float32)
        return tfd.Normal(loc=mean, scale=var)


class Decoder(tf.keras.Model):
    def __init__(self, decoder_net,retain_dec_state=False):
        """ 
        Decoder parent class with no specified output distribution
            :param decoder_net: A tf.keras.Model implementing the decoder arch
            :param retain_dec_state: Boolean - used in RNN based decoders and directs the decoder wether to retain its state between calls
        """
        super(Decoder, self).__init__()
        self.retain_dec_state = retain_dec_state
        self.net = decoder_net

    def __call__(self, x):
        pass


class BernoulliDecoder(Decoder):
    """ Decoder with Bernoulli output distribution (used for HMNIST) """
    def __call__(self, x):
        mapped = self.net(x)
        return tfd.Bernoulli(logits=mapped)


class GaussianDecoder(Decoder):
    """ Decoder with Gaussian output distribution """
    def __call__(self, x, states=None):
        if self.retain_dec_state:
            mean, final_state = self.net(x,states=states) 
            
        else:
            mean = self.net(x)


        var = tf.ones(tf.shape(mean), dtype=tf.float32)
        if self.retain_dec_state:
            return tfd.Normal(loc=mean, scale=var), final_state
        else:
            return tfd.Normal(loc=mean, scale=var)


class gaussian_decoder_learned_variance(Decoder):
    """ Decoder with Gaussian output distribution """
    def __call__(self, x, states=None):
        # get a decoder output with shape (batch_size, seq_len, 2*data_dim)
        if self.retain_dec_state:
            dec_out, final_state = self.net(x,states=states)
        else:
            dec_out = self.net(x)
        # obtain mean and variance
        data_dim = dec_out.shape[2] // 2
        mean = dec_out[:,:,:data_dim]
        var = dec_out[:,:,data_dim:]

        
        if self.retain_dec_state:
            return tfd.Normal(loc=mean, scale=var), final_state
        else:
            return tfd.Normal(loc=mean, scale=var)


# Image preprocessor

class ImagePreprocessor(tf.keras.Model):
    def __init__(self, image_shape, hidden_sizes=(256, ), kernel_size=3.):
        """ Decoder parent class without specified output distribution
            :param image_shape: input image size
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param kernel_size: kernel/filter width and height
        """
        super(ImagePreprocessor, self).__init__()
        self.image_shape = image_shape
        self.net = make_2d_cnn(image_shape[-1], hidden_sizes, kernel_size)

    def __call__(self, x):
        return self.net(x)


# VAE models

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, data_dim, time_length,
                 encoder,
                 decoder, beta=1.0, M=1, K=1, retain_enc_state=False, retain_dec_state=False, latent_space_time_length_down_sample_ratio=1, 
                 **kwargs):
        """ 
        Basic Variational Autoencoder with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            :param encoder: encoder model class
            :param decoder: decoder model class
            
            :param beta: tradeoff coefficient between reconstruction and KL terms in ELBO
            :param M: number of Monte Carlo samples for ELBO estimation
            :param K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)

        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.time_length = time_length
        self.down_sample_ratio = latent_space_time_length_down_sample_ratio
        self.latent_space_time_length = time_length//latent_space_time_length_down_sample_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.prior = None
        self.K = K
        self.M = M
        self.retain_enc_state = retain_enc_state
        self.retain_dec_state = retain_dec_state
        self.encoder_state = None
        self.decoder_state = None
        

    def encode(self, x, states=None):
        x = tf.identity(x)  # in case x is not a Tensor already...
        if self.retain_enc_state:
            enc_out,self.encoder_state = self.encoder(x,states=self.encoder_state)
        else:
            enc_out = self.encoder(x)
        return enc_out

    def decode(self, z, states=None):
        z = tf.identity(z)  # in case z is not a Tensor already...
        if self.retain_dec_state:
            dec_out, self.decoder_state = self.decoder(z,states=self.decoder_state)
        else:
            dec_out = self.decoder(z)
        return dec_out

    def __call__(self, inputs):
        return self.decode(self.encode(inputs).sample()).sample()
        #return self.decode(self.encode(inputs).sample()).mean()

    def reset_states(self):
        self.encoder_state = None
        self.decoder_state = None


    def reconstruct(self, x, return_z_mean=False):
        pz = self.prior
        qz_x = self.encode(x)
        z_mean = qz_x.mean()
        
        px_z = self.decode(z_mean)
        x_hat = px_z.mean()
        # since we already have all the components, compute the KL divergence
        kl = self.kl_divergence(qz_x, pz)
        kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
        # sum the KL over all latent dimensions. we're left with the KL term per window in the batch
        kl = tf.expand_dims(tf.reduce_sum(kl, 1),axis=1)
        if return_z_mean:
            return x_hat, kl, z_mean
        else:
            return x_hat, kl


    def compute_observation_loss(self,x):
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        # in case x is not a Tensor already...
        x = tf.identity(x)
        pz = self._get_prior()
        qz_x = self.encode(x)
        z = qz_x.sample()
        px_z = self.decode(z)
        # get the negative log-likelihood. shape=(batch_size, seq_len, data_dim)
        nll = -px_z.log_prob(x)  
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        # sum the nll over the data dimension
        nll = tf.reduce_sum(nll, 2)
        # compute the KL divergence
        kl = self.kl_divergence(qz_x, pz)
        kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
        # sum the KL over all latent dimensions. we're left with the KL term per window in the batch
        kl = tf.expand_dims(tf.reduce_sum(kl, 1),axis=1)

        elbo = -nll - self.beta * kl
        return -elbo
        



    def generate(self, noise=None, num_samples=1):
        if noise is None:
            noise = tf.random_normal(shape=(num_samples, self.latent_dim))
        return self.decode(noise)
    
    def _get_prior(self):
        if self.prior is None:
            self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim, dtype=tf.float32),
                                                    scale_diag=tf.ones(self.latent_dim, dtype=tf.float32))
        return self.prior

    def compute_nll(self, x, y=None, m_mask=None):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None: y = x

        z_sample = self.encode(x).sample()
        x_hat_dist = self.decode(z_sample)
        nll = -x_hat_dist.log_prob(y)  # shape=(BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            nll = tf.where(m_mask, nll, tf.zeros_like(nll))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(nll)

    def compute_mse(self, x, y=None, m_mask=None, binary=False):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None: y = x

        z_mean = self.encode(x).mean()
        x_hat_mean = self.decode(z_mean).mean()  # shape=(BS, TL, D)
        if binary:
            x_hat_mean = tf.round(x_hat_mean)
        mse = tf.math.squared_difference(x_hat_mean, y)
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            mse = tf.where(m_mask, mse, tf.zeros_like(mse))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(mse)

    def _compute_loss(self, x, m_mask=None, return_parts=False):
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)

        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.tile(m_mask, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
            m_mask = tf.cast(m_mask, tf.bool)

        pz = self._get_prior()
        qz_x = self.encode(x)
        z = qz_x.sample()
        px_z = self.decode(z)

        nll = -px_z.log_prob(x)  # shape=(M*K*BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        if m_mask is not None:
            nll = tf.where(m_mask, tf.zeros_like(nll), nll)  # if not HI-VAE, m_mask is always zeros
        nll = tf.reduce_sum(nll, [1, 2])  # shape=(M*K*BS)

        if self.K > 1:
            kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            weights = -nll - kl  # shape=(M*K*BS)
            weights = tf.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, BS)

            elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
            elbo = tf.reduce_mean(elbo)  # scalar
        else:
            # if K==1, compute KL analytically

            kl = self.kl_divergence(qz_x, pz)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            elbo = -nll - self.beta * kl  # shape=(M*K*BS) K=1
            elbo = tf.reduce_mean(elbo)  # scalar

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            kl = tf.reduce_mean(kl)  # scalar
            return -elbo, nll, kl
        else:
            return -elbo

    def compute_loss(self, x, m_mask=None, return_parts=False):
        del m_mask
        return self._compute_loss(x, return_parts=return_parts)

    def kl_divergence(self, a, b):
        return tfd.kl_divergence(a, b)

    def get_trainable_vars(self):
        self.compute_loss(tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
                          tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
        return self.trainable_variables


class gpdir_vae(VAE):
    def __init__(self, 
                 normal_latent_dim,
                 dirichlet_latent_dim,
                 data_dim, 
                 time_length,
                 encoder,
                 decoder, 
                 beta=1.0, 
                 retain_enc_state=False, 
                 retain_dec_state=False, 
                 latent_space_time_length_down_sample_ratio=1, 
                 kernel="cauchy", 
                 sigma=1., 
                 length_scale=1.0, 
                 kernel_scales=1,
                 dirichlet_prior_alphas=None,
                 *args, 
                 **kwargs):
        # normal prior constants
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        # dirichlet prior constants
        if dirichlet_prior_alphas is None:
            self.dirichlet_prior_alphas = tf.ones(dirichlet_latent_dim)
        else:
            self.dirichlet_prior_alphas = dirichlet_prior_alphas
        # init the priors to None. They will be determined on the first call to the loss function
        self.normal_prior = None
        self.dirichlet_prior = None
        self.normal_latent_dim = normal_latent_dim
        self.dirichlet_latent_dim = dirichlet_latent_dim
        super(gpdir_vae, self).__init__(latent_dim=normal_latent_dim, 
                                        data_dim=data_dim, 
                                        time_length=time_length,
                                        encoder=encoder,
                                        decoder=decoder, 
                                        beta=beta, 
                                        retain_enc_state=retain_enc_state, 
                                        retain_dec_state=retain_dec_state, 
                                        latent_space_time_length_down_sample_ratio=latent_space_time_length_down_sample_ratio)
        
    

    def encode(self, x):
        x = tf.identity(x)
        if self.retain_enc_state:
            dirichlet_dist, normal_dist, self.encoder_state = self.encoder(x,states=self.encoder_state)
        else:
            dirichlet_dist, normal_dist = self.encoder(x)
        return dirichlet_dist, normal_dist

    def decode(self, z):
        num_dim = len(z.shape)
        assert num_dim == 3, "In VAE decode, number of Z dimensions must equal 3"
        # the obtained z sample is of shape [batch_size, z_dim, seq_len]. transpose it to [batch_size, seq_len, z_dim]
        zt = tf.transpose(z, perm=[0,2,1])
        if self.retain_dec_state:
            dec_out, self.decoder_state = self.decoder(zt,states=self.decoder_state)
        else:
            dec_out = self.decoder(zt)
        return dec_out

        
    def sample_z(self, dirichlet_dist, normal_dist, return_mean=False):
        # sample from the normal vectors in the latent space - this sample has shape [batch_size, gz_dim, latent_dim_seq_len]
        if return_mean:
            z_normal = normal_dist.mean()
        else:
            z_normal = normal_dist.sample()

        # sample from the dirichlet distribution in the latent space - this sample has shape [batch_size, dz_dim]
        if return_mean:
            z_dir = dirichlet_dist.mean()
        else:
            z_dir = dirichlet_dist.sample()

        # the shape of the dirichlet sample is [batch_size,dz_dim]. Add a dimension and transpose it to have the Z dimension on axis 1 (just like the sampled normal vectors)
        z_dir = tf.expand_dims(z_dir,axis=1)
        z_dir = tf.transpose(z_dir,perm=[0,2,1])       
        
        # the dirichlet representation is not per time point, but for the entire window. repeat the sample for every temporal point
        z_dir = tf.repeat(z_dir, repeats=z_normal.shape[2], axis=2)

        # concatenate both representations along the z axis
        z = tf.concat([z_normal, z_dir], axis=1)

        return z


    def __call__(self, inputs):

        q_zd_x, q_zn_x = self.encode(x)
        z = self.sample_z(q_zd_x, q_zn_x)
        px_z = self.decode(z)
        return px_z.sample()

    def reconstruct(self, x, return_z_mean=False):
        q_zd_x, q_zn_x = self.encode(x)
        z_mean = self.sample_z(q_zd_x, q_zn_x, return_mean=True)
        
        px_z = self.decode(z_mean)
        x_hat = px_z.mean()
        # since we already have all the components, compute the KL divergence
        kl_n = tf.reduce_sum(self.kl_divergence(q_zn_x, self._get_normal_prior()), 1)
        kl_d = self.kl_divergence(q_zd_x, self._get_dirichlet_prior())
        kl = tf.expand_dims(kl_n + kl_d, axis=1)
        if return_z_mean:
            return x_hat, kl, z_mean
        else:
            return x_hat, kl

    def _compute_loss(self, x, m_mask=None, return_parts=False):
        x = tf.identity(x)
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        q_zd_x, q_zn_x = self.encode(x)
        z = self.sample_z(q_zd_x, q_zn_x)
        px_z = self.decode(z)
        # compute the negative log likelihood. shape=(batch_size, seq_len, data_dim)
        nll = -px_z.log_prob(x)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        # if not HI-VAE, m_mask is always zeros
        if m_mask is not None:
            nll = tf.where(m_mask, tf.zeros_like(nll), nll)
        # sum the NLL along the time and channels (data features) dimensions. remain with shape [batch_size,1]
        nll = tf.reduce_sum(nll, [1, 2])

        
        # compute the normal part KL divergence. shape is [batch_size, gz_dim] (gz_dim is the normal latent space dimension)
        p_zn = self._get_normal_prior()
        kl_n = tf.reduce_sum(self.kl_divergence(q_zn_x, p_zn), 1)
        # compute the dirichlet part KL divergence. shape is [batch_size, 1]
        p_zd = self._get_dirichlet_prior()
        kl_d = self.kl_divergence(q_zd_x, p_zd)

        elbo = -nll - self.beta * (kl_n + kl_d)
        elbo = tf.reduce_mean(elbo)

        if return_parts:
            nll = tf.reduce_mean(nll)
            kl = tf.reduce_mean(kl_n + kl_d)
            return -elbo, nll, kl
        else:
            return -elbo

    def kl_divergence(self, posterior, prior):
        kl = tfd.kl_divergence(posterior, prior)
        # replace Nan values with zeros
        kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
        
        return kl

    def _get_dirichlet_prior(self):
        if self.dirichlet_prior is None:
            self.dirichlet_prior = tfd.Dirichlet(self.dirichlet_prior_alphas)
        return self.dirichlet_prior

    def _get_normal_prior(self):
        if self.normal_prior is None:
            # Compute kernel matrices for each latent dimension
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.latent_space_time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.latent_space_time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.latent_space_time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.latent_space_time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.normal_latent_dim - total
                else:
                    multiplier = int(np.ceil(self.normal_latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = np.concatenate(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.normal_latent_dim
            prior_mean = tf.zeros([self.normal_latent_dim, self.latent_space_time_length], dtype=tf.float32)
            self.normal_prior = tfd.MultivariateNormalTriL(loc=prior_mean, scale_tril=tf.linalg.cholesky(kernel_matrix_tiled))
        return self.normal_prior


class HI_VAE(VAE):
    """ HI-VAE model, where the reconstruction term in ELBO is summed only over observed components """
    def compute_loss(self, x, m_mask=None, return_parts=False):
        return self._compute_loss(x, m_mask=m_mask, return_parts=return_parts)


class GP_VAE(HI_VAE):
    def __init__(self, *args, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, **kwargs):
        """ Proposed GP-VAE model with Gaussian Process prior
            :param kernel: Gaussial Process kernel ["cauchy", "diffusion", "rbf", "matern"]
            :param sigma: scale parameter for a kernel function
            :param length_scale: length scale parameter for a kernel function
            :param kernel_scales: number of different length scales over latent space dimensions
        """
        super(GP_VAE, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        if isinstance(self.encoder, JointEncoder):
            self.encoder.transpose = True

        # Precomputed KL components for efficiency
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None
        self.prior = None

    def decode(self, z, states=None):
        num_dim = len(z.shape)
        assert num_dim == 3, "In VAE decode, number of Z dimensions must equal 3"
        # the obtained z sample is of shape [batch_size, z_dim, seq_len]. transpose it to [batch_size, seq_len, z_dim]
        zt = tf.transpose(z, perm=[0,2,1])
        if self.retain_dec_state:
            dec_out, self.decoder_final_state = self.decoder(zt,states=states)
        else:
            dec_out = self.decoder(zt)
        return dec_out

    def _get_prior(self):
        if self.prior is None:
            # Compute kernel matrices for each latent dimension
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.latent_space_time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.latent_space_time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.latent_space_time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.latent_space_time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = np.concatenate(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.latent_dim

            self.prior = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros([self.latent_dim, self.latent_space_time_length], dtype=tf.float32),
                covariance_matrix=kernel_matrix_tiled)
        return self.prior
    '''
    def kl_divergence(self, a, b):
        """ Batched KL divergence `KL(a || b)` for multivariate Normals.
            See https://github.com/tensorflow/probability/blob/master/tensorflow_probability
                       /python/distributions/mvn_linear_operator.py
            It's used instead of default KL class in order to exploit precomputed components for efficiency
        """

        def squared_frobenius_norm(x):
            """Helper to make KL calculation slightly more readable."""
            return tf.reduce_sum(tf.square(x), axis=[-2, -1])

        def is_diagonal(x):
            """Helper to identify if `LinearOperator` has only a diagonal component."""
            return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorDiag))

        if is_diagonal(a.scale) and is_diagonal(b.scale):
            # Using `stddev` because it handles expansion of Identity cases.
            b_inv_a = (a.stddev() / b.stddev())[..., tf.newaxis]
        else:
            if self.pz_scale_inv is None:
                self.pz_scale_inv = tf.linalg.inv(b.scale.to_dense())
                self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv),
                                             self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

            if self.pz_scale_log_abs_determinant is None:
                self.pz_scale_log_abs_determinant = b.scale.log_abs_determinant()

            a_shape = a.scale.shape
            if len(b.scale.shape) == 3:
                _b_scale_inv = tf.tile(self.pz_scale_inv[tf.newaxis], [a_shape[0]] + [1] * (len(a_shape) - 1))
            else:
                _b_scale_inv = tf.tile(self.pz_scale_inv, [a_shape[0]] + [1] * (len(a_shape) - 1))

            b_inv_a = _b_scale_inv @ a.scale.to_dense()

        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            kl_div = (self.pz_scale_log_abs_determinant - a.scale.log_abs_determinant() +
                      0.5 * (-tf.cast(a.scale.domain_dimension_tensor(), a.dtype) +
                      squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
                      b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))))
        return kl_div
    '''
