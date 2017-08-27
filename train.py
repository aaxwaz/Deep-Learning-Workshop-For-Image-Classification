
"""Train and save the model"""

import tensorflow as tf 
import pandas as pd 
import numpy as np 
import os 
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from configuration import ModelConfig, TrainingConfig
from build_graph import *
from data_util import dataset

FLAGS = None
modelName = 'model_ema'

configModel = ModelConfig()
configTrain = TrainingConfig()

training_loss = []

def train_model_for_one_epoch(iterations, train_x, train_y, model, sess, config, record_train_loss = False):

    num_images = len(train_x)
    for _ in range(iterations):
        # Create a random index.
        idx = np.random.choice(num_images,
                               size=config.batch_size,
                               replace=False)
        batch_x = train_x[idx, :, :, :]
        batch_y = train_y[idx, :]

        if record_train_loss: 
            _, temp_loss = sess.run([model['train_step'], model['loss']], feed_dict={model['x']: batch_x, \
                                                                                     model['y']: batch_y, \
                                                                                     model['keep_prob']:config.keep_rate})
            training_loss.append(temp_loss)
        else:
            sess.run(model['train_step'], feed_dict={model['x']: batch_x, \
                                                     model['y']: batch_y, \
                                                     model['keep_prob']:config.keep_rate})

def generate_prediction(val_x, val_y, model, sess, list_to_pred):

    predictions = sess.run([model[p] for p in list_to_pred], feed_dict={model['x']: val_x, \
                                                                        model['y']: val_y, \
                                                                        model['keep_prob']: 1.0})
    return predictions

def generate_graph(val_y_preds, val_y, save = True):

    total_tasks = val_y.shape[1]

    matplotlib.rcParams['figure.figsize'] = (5.0 * total_tasks, 5.0)

    f, (axes) = plt.subplots(1, total_tasks, sharex=False, sharey=False)

    st = f.suptitle("30-minute interval forecast for each task", fontsize="x-large")

    for c in range(total_tasks):
        axes[c].plot(np.exp(val_y[:, c])-1, color = 'orange', label = 'actual totals')
        axes[c].plot(np.exp(val_y_preds[:, c])-1, color = 'green', label = 'predicted totals')
        axes[c].set_title("Cluster {}".format(c))
        axes[c].legend(loc="upper left")

    if save:
        f.savefig(os.path.join(FLAGS.savedSessionDir, 'tempResult.png'))
    else:
        plt.show()

def get_target_col_names(paramsDumpDir):

    target_cols = pd.read_csv(os.path.join(paramsDumpDir, 'targetColumns'), header = None)
    target_cols = [value[0] for value in target_cols.values.tolist()]

    return target_cols

def main(_):

    # download and load data sets
    alldata = dataset(FLAGS.trainDir)
    alldata.maybe_download_and_extract()
    train_data, _, train_labels = alldata.load_training_data()
    test_data, _, test_labels = alldata.load_test_data()
    class_names = dataset.load_class_names()

    iterations = int(train_data.shape[0] / configTrain.batch_size) # total training iterations in each epoch

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        model = build_graph(configModel)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            epoch = 1
            while epoch <= configTrain.epochs:

                train_model_for_one_epoch(iterations, train_data, train_labels, model, sess, configTrain, record_train_loss = True)

                print("Epoch round ", epoch)
                
                val_loss = generate_prediction(test_data, test_labels, model, sess, ['loss'])[0]
                print("Valiation loss ", val_loss)

                epoch += 1

            plt.plot(training_loss)
            plt.show()

            if not os.path.exists(FLAGS.savedSessionDir):
                os.makedirs(FLAGS.savedSessionDir)
            temp_saver = model['saver']()
            save_path = temp_saver.save(sess, os.path.join(FLAGS.savedSessionDir, modelName))

        print("\n\nTraining done. Model saved: ", os.path.join(FLAGS.savedSessionDir, modelName)) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trainDir',
        type=str,
        default='/home/weimin/workshop/',
        help="""\
        Directory that contains all data.\
        """
    )

    parser.add_argument(
        '--savedSessionDir',
        type=str,
        default='/home/weimin/workshop/savedSessions',
        help="""\
        Directory where your created model / session will be saved.\
        """
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




