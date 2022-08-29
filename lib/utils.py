import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def reduce_logmeanexp(x, axis, eps=1e-5):
    """Numerically-stable (?) implementation of log-mean-exp.
    Args:
        x: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
              reduces all dimensions. Must be in the range
              `[-rank(input_tensor), rank(input_tensor)]`.
        eps: Floating point scalar to avoid log-underflow.
    Returns:
        log_mean_exp: A `Tensor` representing `log(Avg{exp(x): x})`.
    """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.log(tf.reduce_mean(
            tf.exp(x - x_max), axis=axis, keepdims=True) + eps) + x_max


def multiply_tfd_gaussians(gaussians):
    """Multiplies two tfd.MultivariateNormal distributions."""
    mus = [gauss.mean() for gauss in gaussians]
    Sigmas = [gauss.covariance() for gauss in gaussians]
    mu_3, Sigma_3, _ = multiply_gaussians(mus, Sigmas)
    return tfd.MultivariateNormalFullCovariance(loc=mu_3, covariance_matrix=Sigma_3)


def multiply_inv_gaussians(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [n, d]
    lambdas: list of precision matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = int(mus[0].shape[0])
    d_z = int(lambdas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), [-1,d_z,d_z])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bij, bj -> bi", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bij, bj -> bi", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new


def multiply_inv_gaussians_batch(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [..., d]
    lambdas: list of precision matrices with shape [..., d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = mus[0].shape.as_list()[:-1]
    d_z = lambdas[0].shape.as_list()[-1]
    identity_matrix = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(d_z), axis=0), axis=0), batch_size+[1,1])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bcij, bcj -> bci", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bcij, bcj -> bci", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new


def multiply_gaussians(mus, sigmas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of covariance matrices.
    mus: list of mean with shape [n, d]
    sigmas: list of covariance matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(sigmas)
    batch_size = [int(n) for n in mus[0].shape[0]]
    d_z = int(sigmas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), batch_size+[d_z,d_z])
    sigma_new = identity_matrix
    mu_new = tf.zeros((batch_size, d_z))
    for mu, sigma in zip(mus, sigmas):
        sigma_inv = tf.linalg.inv(sigma_new + sigma)
        sigma_prod = tf.matmul(tf.matmul(sigma_new, sigma_inv), sigma)
        mu_prod = (tf.einsum("bij,bj->bi", tf.matmul(sigma, sigma_inv), mu_new)
                   + tf.einsum("bij,bj->bi", tf.matmul(sigma_new, sigma_inv), mu))
        sigma_new = sigma_prod
        mu_new = mu_prod
    lambda_new = tf.linalg.inv(sigma_new)
    return mu_new, sigma_new, lambda_new


def compute_masked_mape(x, y, mask=None, eps=1e-5):
        
    assert x.shape == y.shape, "Input (x) and ground truth (y) should have the same shape"
    if mask is None:
        mask = np.ones_like(x)
    
    # x_masked will be equal to x when mask is 1, and will be equal to y (i.e. ground truth values) when mask is zero
    x_masked = np.where(mask, x, y)
    mape = np.sum(np.abs(y-x_masked)/np.maximum(eps,np.abs(y)))/np.count_nonzero(mask)
    return mape


def compute_masked_mse(x, y, mask=None):
        
    assert x.shape == y.shape, "Input (x) and ground truth (y) should have the same shape"
    if mask is None:
        mask = np.ones_like(x)
    # x_masked will be equal to x when mask is 1, and will be equal to y (i.e. ground truth values) when mask is zero
    x_masked = np.where(mask, x, y)
    squared = (y-x_masked)**2
    mse = np.sum(squared)/np.count_nonzero(mask)
    return mse

def nearest_neighbors_imputation(xm, mask, neighbors_axis, debug=False):
        
    assert xm.shape == mask.shape, "Input (x) and mask should have the same shape"
    assert neighbors_axis < len(xm.shape), "neighbor axis invalid"
    x = xm.copy()
    missing_indices = np.argwhere(mask)
    for i in range(missing_indices.shape[0]):
        missing_idx = tuple(missing_indices[i,:])
        missing_idx_on_axis = missing_idx[neighbors_axis]
        if missing_idx_on_axis == 0:
            imputer_idx = missing_idx[:neighbors_axis] + (1,) + missing_idx[neighbors_axis+1:]
            x[missing_idx] = x[imputer_idx]
            if i < 100 and debug:
                print("miss",missing_idx)
                print("imputer",imputer_idx)
                print("\n")
        elif missing_idx_on_axis == x.shape[neighbors_axis]-1:
            imputer_idx = missing_idx[:neighbors_axis] + (-1,) + missing_idx[neighbors_axis+1:]
            x[missing_idx] = x[imputer_idx]
            if i < 100 and debug:
                print("miss",missing_idx)
                print("imputer",imputer_idx)
                print("\n")
        else:
            imputer_idx_forward  = missing_idx[:neighbors_axis] + (missing_idx_on_axis-1,) + missing_idx[neighbors_axis+1:]
            imputer_idx_backward = missing_idx[:neighbors_axis] + (missing_idx_on_axis+1,) + missing_idx[neighbors_axis+1:]
            x[missing_idx] = (x[imputer_idx_forward] + x[imputer_idx_backward])/2
            if i < 100 and debug:
                print("miss",missing_idx)
                print("imputer_b",imputer_idx_backward)
                print("imputer_f",imputer_idx_forward)
                print("\n")
        
            
    return x


def prepare_smd_data_for_imputation_experiment(series_dict, window_size, percent_missing = 0.2, validation_set_size=0.1):
    all_files_full_train = []
    all_files_full_val = []
    all_files_full_test = []
    all_files_miss_train = []
    all_files_miss_val = []
    all_files_miss_test = []
    all_files_mask_train = []
    all_files_mask_val = []
    all_files_mask_test = []
    for key in series_dict.keys():
        series = series_dict[key]['train']
        print(f"SMD train data shape of {key} is {series.shape}")
        full_train, full_val, full_test, miss_train, miss_val, miss_test, mask_train, mask_val, mask_test = prepare_data_for_imputation_experiment(series, 
            window_size, 
            percent_missing=percent_missing, 
            validation_set_size=validation_set_size)
        if full_train is not None:
            all_files_full_train.append(full_train)
        if full_val is not None:
            all_files_full_val.append(full_val)
        if full_test is not None:
            all_files_full_test.append(full_test)
        if miss_train is not None:
            all_files_miss_train.append(miss_train)
        if miss_val is not None:
            all_files_miss_val.append(miss_val)
        if miss_test is not None:
            all_files_miss_test.append(miss_test)
        if mask_train is not None:
            all_files_mask_train.append(mask_train)
        if mask_val is not None:
            all_files_mask_val.append(mask_val)
        if mask_test is not None:
            all_files_mask_test.append(mask_test)
    if len(all_files_full_train) > 0:
        full_train = np.concatenate(all_files_full_train)
    if len(all_files_full_val) > 0:
        full_val = np.concatenate(all_files_full_val)
    if len(all_files_full_test) > 0:
        full_test = np.concatenate(all_files_full_test)
    if len(all_files_miss_train) > 0:
        miss_train = np.concatenate(all_files_miss_train)
    if len(all_files_miss_val) > 0:
        miss_val = np.concatenate(all_files_miss_val)
    if len(all_files_miss_test) > 0:
        miss_test = np.concatenate(all_files_miss_test)
    if len(all_files_mask_train) > 0:
        mask_train = np.concatenate(all_files_mask_train)
    if len(all_files_mask_val) > 0:
        mask_val = np.concatenate(all_files_mask_val)
    if len(all_files_mask_test) > 0:
        mask_test = np.concatenate(all_files_mask_test)
    return full_train, full_val, full_test, miss_train, miss_val, miss_test, mask_train, mask_val, mask_test

    


def prepare_data_for_imputation_experiment(series, window_size, percent_missing = 0.2, validation_set_size=0.1,return_dict=False):
    assert series.shape[0] > window_size, "time series length must be greater then the window size"
    # first, truncate the data size (in time domain) such that the number of samples in an integer multiple of the window size
    new_size = series.shape[0] - (series.shape[0] % window_size)
    full_data = series[:new_size,:]
    # reshape the data 
    full_data_reshaped = np.reshape(full_data,(new_size//window_size,window_size,full_data.shape[1]))
    print("orig data shape is ",full_data.shape)
    print("new data shape is ",full_data_reshaped.shape)

    # now find which windows have Nan values in them and remove them from the data set
    nan_indices = np.argwhere(np.isnan(full_data_reshaped))
    windows_with_nan = np.unique(nan_indices[:,0])
    if nan_indices.size > 0:
        # remove these windows
        full_data_reshaped = np.delete(full_data_reshaped, windows_with_nan, axis=0)
        print(f"removed {len(windows_with_nan)} windows with Nan values")
        print("new data shape is ",full_data_reshaped.shape)
    else:
        print("no null entries found")

    # force missing values in the data by randomly setting values to zeros. do it first by creating a random mask indicating where the missing values are
    mask = np.random.uniform(size = full_data_reshaped.shape) < percent_missing
    print("percent of missing values is ",np.sum(mask)/mask.size)
    # null some values according to the mask
    miss_data = full_data_reshaped.copy()
    miss_data[mask] = 0

    # just a sanity check that the places where missing data was forced are indeed indicated by the mask. Since "missing" data is basically
    # forcing some entries to zero, we examine only mask entries corresponding to non-zero entries in the full data
    ne = (miss_data != full_data_reshaped)
    diff_from_zero = (full_data_reshaped != 0)
    assert np.allclose(ne,np.logical_and(mask,diff_from_zero))
    # split the data into two sets: train and validation+test
    if validation_set_size > 0:
        full_data_train, full_data_val_test, miss_data_train, miss_data_val_test, mask_train, mask_val_test = train_test_split(full_data_reshaped, miss_data, mask, test_size=validation_set_size)
        # split the val_test set into validation and test sets
        full_data_val, full_data_test, miss_data_val, miss_data_test, mask_val, mask_test = train_test_split(full_data_val_test, miss_data_val_test, mask_val_test, test_size=0.5)
    else:
        full_data_train = full_data_reshaped
        miss_data_train = miss_data
        mask_train = mask
        full_data_val = None
        full_data_test = None
        miss_data_val = None
        miss_data_test = None
        mask_val = None
        mask_test = None
    if return_dict:
        return {"full_data_train" : full_data_train, 
                "full_data_val" : full_data_val, 
                "full_data_test" : full_data_test, 
                "miss_data_train" : miss_data_train, 
                "miss_data_val" : miss_data_val, 
                "miss_data_test" : miss_data_test, 
                "mask_train" : mask_train, 
                "mask_val" : mask_val, 
                "mask_test" : mask_test}
    else:
        return full_data_train, full_data_val, full_data_test, miss_data_train, miss_data_val, miss_data_test, mask_train, mask_val, mask_test

def train_loop(model, 
               num_epochs, 
               train_ds, 
               val_ds, 
               optimizer, 
               checkpoint, 
               checkpoint_prefix, 
               gradient_clip=1e3,
               log_every=10, 
               verbose=False, 
               train_for_imputation = True, 
               use_sequential_ds=False, 
               batch_size_low=32, 
               batch_size_high=64):
    trainable_vars = model.get_trainable_vars()
    losses_train = []
    kl_term_train = []
    nll_train = []
    train_mse = []
    val_mse_list = []
    min_val_mse = np.inf
    
    for epoch in range(num_epochs):
        acc_elbo = 0
        acc_kl = 0
        acc_nll = 0
        model.reset_states()
        if use_sequential_ds:
            batch_size = np.random.randint(low=batch_size_low,high=batch_size_high)
            train_ds.pick_data(batch_size)
            iterator = train_ds.iterator()
        else:
            iterator = train_ds
        # Iterate over the batches of the dataset.
        for step, batch in enumerate(iterator):
            if train_for_imputation:
                (x_gt_seq, x_seq, m_seq) = batch
            else:
                x_seq = batch
                m_seq = None
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Ensures that the trainable parameters are being traced by this tape.
                tape.watch(trainable_vars)
                # compute the loss for this batch - this include also a forward pass in the model
                loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, trainable_vars)
            grads = [np.nan_to_num(grad) for grad in grads]
            grads, global_norm = tf.clip_by_global_norm(grads, gradient_clip)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, trainable_vars))
            acc_elbo += loss.numpy()
            acc_kl += kl.numpy()
            acc_nll += nll.numpy()
            losses_train.append(loss.numpy())
            kl_term_train.append(kl.numpy())
            nll_train.append(nll.numpy())
            # Log every n batches.
            if step % log_every == 0:
                train_batch_mse = None
                val_mse = None
                if train_for_imputation:
                    n_missings = tf.math.reduce_sum(m_seq)
                    train_batch_mse = model.compute_mse(x=x_seq, y=x_gt_seq, m_mask=m_seq) / n_missings
                    train_mse.append(train_batch_mse)
                
                if train_for_imputation and val_ds is not None :
                    # calculate the validation MSE
                    val_missing = 0
                    val_missing_sq_error = 0
                    for (val_gt_seq, val_miss_seq, val_mask_seq) in val_ds:
                        # update the number of missing values in the validation set
                        val_missing += tf.math.reduce_sum(val_mask_seq).numpy()
                        val_missing_sq_error += model.compute_mse(x=val_miss_seq, y=val_gt_seq, m_mask=val_mask_seq).numpy()
                    val_mse = val_missing_sq_error/val_missing
                    
                    val_mse_list.append(val_mse)
                    if val_mse < min_val_mse:
                        min_val_mse = val_mse
                        checkpoint.write(checkpoint_prefix)
                if verbose:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss.numpy())))
                    if train_batch_mse is not None: print("Training MSE (for one batch) at step %d: %.4f" % (step, train_batch_mse))
                    if val_mse is not None: print(f"Validation MSE is {val_mse}")
        print(f"in epoch {epoch+1} mean elbo is {acc_elbo/(step+1)}")
        print(f"in epoch {epoch+1} mean kl is {acc_kl/(step+1)}")
        print(f"in epoch {epoch+1} mean nll is {acc_nll/(step+1)}")
        print("-------------------------------------------------------------------------------------")
    return losses_train, kl_term_train, nll_train, train_mse, val_mse_list






def calculate_f1(labels,tpr,fpr,thresholds):
    assert tpr.shape == fpr.shape, "tpr & fpr must have the same shape"
    # find thresholds for which tpr & fpr are both zeros
    tpr_zero = np.argwhere(tpr == 0).squeeze()
    fpr_zero = np.argwhere(fpr == 0).squeeze()
    both_zero = np.intersect1d(tpr_zero,fpr_zero)
    tpr = np.delete(tpr,tpr_zero)
    fpr = np.delete(fpr,tpr_zero)
    thresholds = np.delete(thresholds,tpr_zero)
    # find how many positive examples are in the data
    P = np.count_nonzero(labels)
    # and negative ones...
    N = labels.size - P
    # calculate the number of true positives from: recall = TPR = TP/P --> TP = P*TPR
    TP = P * tpr
    # calculate the number of false positives from: FPR = FP/N --> FP = N*FPR
    FP = N * fpr
    # and now we can calculate the precision = TP/(TP+FP)
    precision = TP/(TP + FP)
    recall = tpr
    f1 = (2 * recall * precision)/(recall + precision)
    return f1, recall, precision, thresholds


def get_reconstruction(model, data, labels, seq_len, batch_size=32, return_z_mean=False, repeat_kl=None, transpose_z=True):
    if repeat_kl is not None:
        repeat_kl = repeat_kl
    else:
        repeat_kl = seq_len
    if model.retain_enc_state or model.retain_dec_state:
        # truncate the data size (in time domain) such that the number of samples in an integer multiple of (seq_len*batch_size)
        new_size = data.shape[0] - (data.shape[0] % (seq_len*batch_size))
        data = data[:new_size,:]
        labels = labels[:new_size]
        print(f"slicing the series data to sub-sequences with length {new_size/batch_size}")
        sampler = sequence_sampler(data=data,seq_len=seq_len,batch_size=batch_size)
        sampler.pick_data(batch_size)
        print(f"number of batches in the sampler is {sampler.num_batches}")
        iterator = sampler.iterator()
        # reset the model's internal states
        model.reset_states()
        
    else:
        # truncate the data size (in time domain) such that the number of samples in an integer multiple of the window size
        print(f"slicing the series data to windows len {seq_len}")
        new_size = data.shape[0] - (data.shape[0] % seq_len)
        data = data[:new_size,:]
        # extract the labels corresponding to the data we kept
        labels = labels[:new_size]
        # reshape the data 
        data_reshaped = np.reshape(data,(new_size//seq_len,seq_len,data.shape[1]))
        print("orig data shape is ",data.shape)
        print("new data shape is " ,data_reshaped.shape)
        iterator = tf.data.Dataset.from_tensor_slices(data_reshaped).batch(batch_size)
       
    # using the trained model, reconstruct the data, and for each window get the KL term as well
    data_reconstructed_batches = []
    kl_batches = []
    z_mean_batches = []
    for batch_idx, batch in enumerate(iterator):
        if return_z_mean:
            x_hat, kl, z_mean = model.reconstruct(x=batch, return_z_mean=True)
            if transpose_z:
                # z_mean shape is [batch_size, z_dim, latent_space_seq_len]. transpose to [batch_size, latent_space_seq_len, z_dim]
                z_mean = tf.transpose(z_mean, perm=[0,2,1])
            
            z_mean_batches.append(z_mean.numpy())
        else:
            x_hat, kl = model.reconstruct(x=batch)
        data_reconstructed_batches.append(x_hat.numpy())
        # we may have a single KL term for each window in the batch. 
        # in that case repeat it seq_len times such that each time point will have that term
        # we may also have a KL term for each latent variable - in this case the repeat factor will be given
        # on function call. in any case we repeat the KL term here to assign each data point its KL term
        kl = np.repeat(kl.numpy(), repeat_kl, axis=1)
        kl_batches.append(kl)
        
    if model.retain_enc_state or model.retain_dec_state:
        # concatenate the batches along the time axis
        data_reconstructed = np.concatenate(data_reconstructed_batches,axis=1)
        kl = np.concatenate(kl_batches,axis=1)
        if len(z_mean_batches) > 0:
            z_mean = np.concatenate(z_mean_batches,axis=1)

    else:
        # concatenate the batches to obtain a long and chronologically ordered array of windows
        data_reconstructed = np.concatenate(data_reconstructed_batches)
        kl = np.concatenate(kl_batches)
        if len(z_mean_batches) > 0:
            z_mean = np.concatenate(z_mean_batches)

    # reshape the original and reconstructed data
    print("reshaping the reconstructed data to the original shapes")
    data_reconstructed = np.reshape(data_reconstructed, (data_reconstructed.shape[0]*data_reconstructed.shape[1],data_reconstructed.shape[2]))
    kl_reconstructed = np.reshape(kl, (kl.shape[0]*kl.shape[1],))
    print("data_reconstructed.shape = ",data_reconstructed.shape)
    print("kl_reconstructed.shape = ",kl_reconstructed.shape)
    print("data_orig.shape = ",data.shape)
    if return_z_mean:
        z_mean = np.reshape(z_mean, (z_mean.shape[0]*z_mean.shape[1],z_mean.shape[2]))
        print("z_mean.shape = ", z_mean.shape)
        return data, data_reconstructed, kl_reconstructed, labels, z_mean
    else:
        return data, data_reconstructed, kl_reconstructed, labels


def get_reconstruction_error(model, 
                             data, 
                             labels, 
                             seq_len, 
                             batch_size=32, 
                             stateful_encoder=False,
                             stateful_decoder=False, 
                             repeat_kl=None, 
                             transpose_z=True):

    data, data_reconstructed, kl_reconstructed, labels = get_reconstruction(model, 
                                                                            data, 
                                                                            labels, 
                                                                            seq_len, 
                                                                            batch_size=batch_size, 
                                                                            repeat_kl=repeat_kl, 
                                                                            transpose_z=transpose_z)
    # calculate the reconstruction MSE
    print("calculating reconstruction MSE")
    reconstruct_error = np.linalg.norm(data_reconstructed-data,axis=1)
    print("reconstruct_error.shape = ", reconstruct_error.shape)

    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=reconstruct_error/np.max(reconstruct_error))
    auc = roc_auc_score(y_true=labels, y_score=reconstruct_error/np.max(reconstruct_error))
    print(f"reconstruction error based AUC is {auc}")

    
    recons_plus_kl = reconstruct_error + kl_reconstructed
    recons_plus_kl_fpr, recons_plus_kl_tpr, recons_plus_kl_thresholds = roc_curve(y_true=labels, y_score=recons_plus_kl/np.max(recons_plus_kl))
    recons_plus_kl_auc = roc_auc_score(y_true=labels, y_score=recons_plus_kl/np.max(recons_plus_kl))
    


    # separate the reconstruction errors of the "normal" measurements and the "abnormal" ones
    normal_labels_idx   = np.nonzero(labels == 0)
    abnormal_labels_idx = np.nonzero(labels == 1)
    normal_reconst_error   = reconstruct_error[normal_labels_idx]
    abnormal_reconst_error = reconstruct_error[abnormal_labels_idx]

    normal_reconst_error_plus_kl   = recons_plus_kl[normal_labels_idx]
    abnormal_reconst_error_plus_kl = recons_plus_kl[abnormal_labels_idx]
    
    f1, recall, precision, thresholds = calculate_f1(labels,tpr,fpr,thresholds)
    max_f1 = np.max(f1)
    print(f"F1 max is {max_f1}")
    
    fig,ax = plt.subplots(figsize=(12,6))
    ax.hist(normal_reconst_error,bins=100,label="normal")
    ax.hist(abnormal_reconst_error,bins=100,label="anomaly")
    ax.legend()
    ax.set_xlabel('error magnitude')
    ax.set_ylabel('count')
    ax.set_title(f"reconstruction errors histograms")

    plt.show()
    fig,ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(fpr,tpr,label="reconstruction based")
    ax[0].set_xlabel('fpr')
    ax[0].set_ylabel('tpr')
    ax[0].set_title(f"ROC curve for anomaly detection")
    ax[0].legend()
    ax[1].plot(thresholds,recall,label='recall')
    ax[1].plot(thresholds,precision,label='precision')
    ax[1].set_xlabel('thresholds')
    ax[1].set_ylabel('prec/recall')
    ax[1].set_title(f"precision and recall for anomaly detection")
    ax[1].legend()
    plt.show()
    results_dict = {'fpr'                       : fpr, 
                    'tpr'                       : tpr, 
                    'thresholds'                : thresholds, 
                    'auc'                       : auc, 
                    'recons_plus_kl_fpr'        : recons_plus_kl_fpr, 
                    'recons_plus_kl_tpr'        : recons_plus_kl_tpr, 
                    'recons_plus_kl_thresholds' : recons_plus_kl_thresholds, 
                    'recons_plus_kl_auc'        : recons_plus_kl_auc, 
                    'max_f1'                    : max_f1, 
                    'recall'                    : recall, 
                    'precision'                 : precision, 
                    'reconstruct_error'         : reconstruct_error}
    return results_dict


def get_perf_from_scores(scores,labels):
    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=scores)
    auc = roc_auc_score(y_true=labels, y_score=scores)
    print(f"AUC is {auc}")
    f1, recall, precision, thresholds = calculate_f1(labels,tpr,fpr,thresholds)
    max_f1 = np.max(f1)
    print(f"F1 best is {max_f1}")
    # separate the scores of normal and anomalies
    normal_scores_idx  = np.nonzero(labels == 0)
    anomaly_scores_idx = np.nonzero(labels >= 1)
    normal_scores  = scores[normal_scores_idx]
    anomaly_scores = scores[anomaly_scores_idx]
    # plot scores
    fig,ax = plt.subplots(figsize=(12,6))
    ax.hist(normal_scores,bins=100,label="normal")
    ax.hist(anomaly_scores,bins=100,label="anomaly")
    ax.legend()
    ax.set_xlabel('score')
    ax.set_ylabel('count')
    ax.set_title(f"anomaly scores histograms")
    plt.show()
    

    return auc, max_f1

def observe_latent_variables_means_distributions(means,labels):
    z_dim = means.shape[1]
    # separate the reconstruction errors of the "normal" measurements and the "abnormal" ones
    normal_labels_idx = np.nonzero(labels == 0)
    normal_z_means = np.squeeze(means[normal_labels_idx,:])
    abnormal_labels_idx = np.nonzero(labels == 1)
    abnormal_z_means = np.squeeze(means[abnormal_labels_idx,:])
    
    for i in range(z_dim):
        fig,ax = plt.subplots(figsize=(6,6))
        ax.hist(normal_z_means[:,i],bins=100,label="normal")
        ax.hist(abnormal_z_means[:,i],bins=100,label="abnormal")
        ax.legend()
        ax.set_xlabel('latent variable mean')
        ax.set_ylabel('count')
        ax.set_title(f"latent variable z_{i} means histograms")
        plt.show()

def embed_latent_representation(z_mean, labels, sample_size, z_splits, titles):

    if sample_size < z_mean.shape[0]:
        smpl_idx = np.random.choice(np.arange(z_mean.shape[0]), size=sample_size, replace=False)
        z_mean = z_mean[smpl_idx,:]
        labels = labels[smpl_idx]
    
    n_idx  = np.nonzero(labels == 0)
    an_idx = np.nonzero(labels >= 1)
    split_boundaries = [0] + z_splits + [z_mean.shape[1]]
    
    num_splits = len(split_boundaries)-1
    fig,ax = plt.subplots(1,num_splits,figsize=(10*num_splits,10))
    for i in range(num_splits):
        split = z_mean[:,split_boundaries[i]:split_boundaries[i+1]]
        split_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=50).fit_transform(split)
        if num_splits == 1:
            axh = ax
        else:
            axh = ax[i]
        axh.scatter(split_embedded[n_idx,0]  ,split_embedded[n_idx,1]  ,marker='.',label='normal')
        axh.scatter(split_embedded[an_idx,0] ,split_embedded[an_idx,1] ,marker='.',label='anomaly')
        axh.set_xlabel('x1')
        axh.set_ylabel('x2')
        axh.set_title(titles[i])
        axh.legend()
    plt.show()
    return

class sequence_sampler():
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """
    def __init__(self, data, seq_len, batch_size):
        
        # check the parameters
        if seq_len < 1:
            raise ValueError('seq_len must be at least 1')
        self.data_len = data.shape[0]
        if self.data_len < seq_len:
            raise ValueError("number of samples must be at least as large as seq_len")
        self.batch_size = None
        self.seq_len = seq_len
        self.data_dim = data.shape[1]
        self.orig_data = data

        
    def pick_data(self,batch_size):
        self.batch_size = batch_size
        new_size = (self.data_len // self.seq_len) * self.seq_len
        if new_size == self.data_len:
            self.start_timestep = 0
        else:    
            self.start_timestep = np.random.randint(low=0, high=(self.data_len % self.seq_len))
        self.data = self.orig_data[self.start_timestep:self.start_timestep+new_size,:]
        # reshape data to sequences
        self.data = np.reshape(self.data,newshape=(-1,self.seq_len,self.data.shape[1]))
        # collect the batches starting from a random sequence
        if (self.data.shape[0] % self.batch_size) == 0:
            self.start_seq = 0
        else:
            self.start_seq = np.random.randint(low=0, high=(self.data.shape[0] % self.batch_size))
        self.num_batches = (self.data.shape[0]-self.start_seq) // self.batch_size



    def iterator(self):
               
        for i in range(self.num_batches):
            yield np.stack([self.data[i + self.start_seq + j * self.num_batches].astype(np.float32) for j in range(self.batch_size)])

    def unit_test(self, orig_data, plot=False):
        self.pick_data()
        sequential_batches = []
        for step, batch in enumerate(self.iterator()):
            sequential_batches.append(batch)

        # concatenate the batches along the time axis
        sequences = np.concatenate(sequential_batches,axis=1)        
        print("sequences.shape = ",sequences.shape)
        # just a quick sanity check of sizes
        assert sequences.shape[0] == self.batch_size, f"There should be {self.batch_size} sequences"
        assert sequences.shape[2] == self.data_dim, "Data dimension doesn't match"

        for i in range(self.batch_size):
            seq_start = self.start_timestep + self.start_seq * self.seq_len + i * self.num_batches * self.seq_len
            seq_end = seq_start + sequences.shape[1]
            orig_sequence = orig_data[seq_start:seq_end]
            sequence = np.squeeze(sequences[i])
            if plot:
                channel = np.random.randint(low=0,high=self.data_dim)
                fig,ax = plt.subplots(figsize=(6,6))
                ax.plot(orig_sequence[:,channel],label="orig")
                ax.plot(sequence[:,channel],'*r',label="batched")
                ax.legend()
                ax.set_xlabel('timestep')
                ax.set_ylabel('y')
                ax.set_title(f"orig and batched sequences in channel {channel}")
                ax.legend()
                plt.show()
            if not np.array_equal(orig_sequence, sequence):
                print("batched sequence.shape = ",sequence.shape)
                print("orig sequence.shape = ",orig_sequence.shape)
                print(sequence)
                print(orig_sequence)
            assert np.array_equal(orig_sequence, sequence), f"in sub sequence {i}, batched sub-sequence does not match the original sequences"
    
