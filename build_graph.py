import tensorflow as tf 
import numpy as np 

def build_graph(config):
    """This function builds the graph for a deep net for classifying images.
    Args:
      config: Model configuration object
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the image into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """

    x_image = tf.placeholder(tf.float32, [None, config.image_height, config.image_width, config.image_channels])
    y = tf.placeholder(tf.float32, [None, int(config.num_classes)])
    is_training = tf.placeholder(tf.bool, [])
    keep_prob = tf.placeholder(tf.float32)

    # First convolutional module [conv-conv-pool] -- maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1_1'):
        W_conv1_1 = weight_variable([config.filter_size, config.filter_size, config.image_channels, config.conv1_num_filters], name = 'W')
        b_conv1_1 = bias_variable([config.conv1_num_filters], name = 'b')
        c_conv1_1 = conv2d(x_image, W_conv1_1) + b_conv1_1
        #c_conv1_1 = batch_norm1(c_conv1_1, is_training)
        h_conv1_1 = tf.nn.relu(c_conv1_1)

    with tf.name_scope('conv1_2'):
        W_conv1_2 = weight_variable([config.filter_size, config.filter_size, config.conv1_num_filters, config.conv1_num_filters], name = 'W')
        b_conv1_2 = bias_variable([config.conv1_num_filters], name = 'b')
        c_conv1_2 = conv2d(h_conv1_1, W_conv1_2) + b_conv1_2
        #c_conv1_2 = batch_norm1(c_conv1_2, is_training)
        h_conv1_2 = tf.nn.relu(c_conv1_2)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1_2)

    # Second convolutional module [conv-conv-pool] -- maps 32 feature maps to 64.
    with tf.name_scope('conv2_1'):
        W_conv2_1 = weight_variable([config.filter_size, config.filter_size, config.conv1_num_filters, config.conv2_num_filters], name = 'W')
        b_conv2_1 = bias_variable([config.conv2_num_filters], name = 'b')
        c_conv2_1 = conv2d(h_pool1, W_conv2_1) + b_conv2_1
        #c_conv2_1 = batch_norm1(c_conv2_1, is_training)
        h_conv2_1 = tf.nn.relu(c_conv2_1)

    with tf.name_scope('conv2_2'):
        W_conv2_2 = weight_variable([config.filter_size, config.filter_size, config.conv2_num_filters, config.conv2_num_filters], name = 'W')
        b_conv2_2 = bias_variable([config.conv2_num_filters], name = 'b')
        c_conv2_2 = conv2d(h_conv2_1, W_conv2_2) + b_conv2_2
        #c_conv2_2 = batch_norm1(c_conv2_2, is_training)
        h_conv2_2 = tf.nn.relu(c_conv2_2)
        
    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2_2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
    # is down to 4x4x64 feature maps -- maps this to 1024 features.
    #feature_map_flattened_dim = int((config.image_height/(2**2)) * (config.image_width/(2**2)) * config.conv2_num_filters)
    feature_map_flattened_dim = int(np.prod(h_pool2.get_shape()[1:]))

    h_pool2_flat = tf.reshape(h_pool2, [-1, feature_map_flattened_dim])
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([feature_map_flattened_dim, config.fc1_num_features], name = 'W')
        b_fc1 = bias_variable([config.fc1_num_features], name = 'b')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('dropout1'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """with tf.name_scope('fc2'):
        W_fc2 = weight_variable([config.fc1_num_features, config.fc2_num_features], name = 'W')
        b_fc2 = bias_variable([config.fc2_num_features], name = 'b')
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    #h_fc2 = batch_norm1(h_fc2, is_training)
    h_fc2 = tf.nn.relu(h_fc2)
    with tf.name_scope('dropout2'):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)"""

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([config.fc1_num_features, config.num_classes], name = 'W')
        b_fc3 = bias_variable([config.num_classes], name = 'b')
    
    # Raw predictions - logits
    with tf.name_scope('logits'):
        logits = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

    with tf.name_scope('probabilities'):
        probs = tf.nn.softmax(logits)

    with tf.name_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                       logits=logits)
    loss = tf.reduce_mean(loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver

    # Return the model in dict
    return dict(
        x_image = x_image, 
        y = y, 
        is_training = is_training, 
        keep_prob = keep_prob, 
        h_conv1_1 = h_conv1_1,
        logits = logits, 
        probs = probs, 
        loss = loss, 
        train_step = train_step, 
        accuracy = accuracy, 
        saver = saver, 
        )

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_strided(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def batch_norm1(h, is_training):
    return tf.contrib.layers.batch_norm(h, 
                                        center=True, 
                                        scale=True, 
                                        is_training=is_training)

def weight_variable(shape, name = None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name = None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name = name)

def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed