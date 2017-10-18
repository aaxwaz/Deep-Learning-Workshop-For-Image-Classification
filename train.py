
"""Train and save the model"""

import tensorflow as tf 
import numpy as np 
import os 
import argparse
import sys
from datetime import datetime 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from configuration import ModelConfig, TrainingConfig
from build_graph import *
from data_utils import dataset
from vis_utils import visualize_grid

FLAGS = None
modelName = 'model_ema'
PLOT_WEIGHTS_EVERY_EPOCH = 5

configModel = ModelConfig()
configTrain = TrainingConfig()

training_loss = []
validation_loss = []
validation_accu = []

def train_model_for_one_epoch(iterations, train_x, train_y, model, sess, config, record_train_loss = False):

    num_images = len(train_x)
    for i in range(iterations):
        # Create a random index.
        idx = np.random.choice(num_images,
                               size=config.batch_size,
                               replace=False)
        batch_x = train_x[idx, :, :, :]
        batch_y = train_y[idx, :]

        
        _, temp_loss = sess.run([model['train_step'], model['loss']], feed_dict={model['x_image']: batch_x, \
                                                                                     model['y']: batch_y, \
                                                                                     model['is_training']: True, \
                                                                                     model['keep_prob']:config.keep_rate})
        if record_train_loss: 
            training_loss.append(temp_loss)



def generate_prediction(val_x, val_y, model, sess, list_to_pred):

    predictions = sess.run([model[p] for p in list_to_pred], feed_dict={model['x_image']: val_x, \
                                                                        model['y']: val_y, \
                                                                        model['is_training']: False, \
                                                                        model['keep_prob']: 1.0})
    return predictions

def plot_training_loss(train_loss, saveName):

    plt.title('Training loss')
    plt.plot(train_loss)
    plt.xlabel('No. of training iteractions')
    plt.ylabel('Training losses')
    plt.savefig(saveName)
    plt.close()
    print("training losses saved at: ", saveName)

def plot_val_loss_n_accuracy(validation_loss, validation_accu, saveName):

    fig, ax1 = plt.subplots()
    ax1.plot(range(len(validation_loss)), validation_loss, 'b', )
    ax1.set_xlabel('No. of training epochs')
    ax1.set_ylabel('Val losses', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(range(len(validation_accu)), validation_accu, 'g')
    ax2.set_ylabel('Val accuracy', color='g')
    ax2.tick_params('y', colors='g')
    fig.tight_layout()
    plt.savefig(saveName)
    plt.close()
    print("Validation losses and accuracy are saved: ", saveName)

def vis_activations_from_model(train_data, model, sess, save_dir, seed=10):

    np.random.seed(seed)
    idx = np.random.choice(train_data.shape[0],
          size=1,
          replace=False)
    img = train_data[idx[0], :, :, :]
    ## visualize layer-1 kernel weights in grid 
    img = np.expand_dims(img, 0)
    h_conv1_1 = sess.run(model['h_conv1_1'], feed_dict={model['x_image']:img})
    h_conv1_1 = h_conv1_1.transpose(3, 1, 2, 0)   # reshape to: (N, H, W, 1)
    vis_grid = visualize_grid(h_conv1_1, grey = True)
    plot_weights_in_grid(vis_grid, os.path.join(save_dir, 'vis_activations.png'))

def plot_weights_in_grid(vis_grid, saveName, gray = True):

    if gray:
        plt.imshow(vis_grid.astype('uint8'), cmap = 'gray')
    else:
        plt.imshow(vis_grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.savefig(saveName)
    plt.close()
    print("Visualization is saved at: ", saveName)

def main(_):

    # download and load data sets
    alldata = dataset(FLAGS.trainDir)
    alldata.maybe_download_and_extract()
    train_data, _, train_labels = alldata.load_training_data()
    test_data, _, test_labels = alldata.load_test_data()
    class_names = alldata.load_class_names()

    iterations = int(train_data.shape[0] / configTrain.batch_size) # total training iterations in each epoch

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        model = build_graph(configModel)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            ## start training epochs
            epoch = 1
            while epoch <= configTrain.epochs:

                now = datetime.now()
                train_model_for_one_epoch(iterations, train_data, train_labels, model, sess, configTrain, record_train_loss = True)
                used_time = datetime.now() - now

                print("\nEpoch round ", epoch, ' used {0} seconds. '.format(used_time.seconds))
                
                val_loss, val_accuracy = generate_prediction(test_data, test_labels, model, sess, ['loss', 'accuracy'])
                validation_loss.append(val_loss)
                validation_accu.append(val_accuracy)
                print("Valiation loss ", val_loss, " and accuracy ", val_accuracy)

                ## if required, visualize activations from image 
                if configTrain.vis_weights_every_epoch > 0 and epoch % configTrain.vis_weights_every_epoch == 0:

                    vis_activations_from_model(train_data, model, sess, FLAGS.trainDir, 10)

                epoch += 1

            ## upon training done, plot training & validation losses and validation accuracy
            print("training done.")
            plot_training_loss(training_loss, os.path.join(FLAGS.trainDir, 'train_losses.png'))
            plot_val_loss_n_accuracy(validation_loss, validation_accu, os.path.join(FLAGS.trainDir, 'val_losses_n_accuracy.png'))

            ## save trained session
            if not os.path.exists(FLAGS.savedSessionDir):
                os.makedirs(FLAGS.savedSessionDir)
            temp_saver = model['saver']()
            save_path = temp_saver.save(sess, os.path.join(FLAGS.savedSessionDir, modelName))

        print("\nTraining done. Model saved: ", os.path.join(FLAGS.savedSessionDir, modelName)) 

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
        default='/home/weimin/workshop/savedSessions/',
        help="""\
        Directory where your created model / session will be saved.\
        """
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




