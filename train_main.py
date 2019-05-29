#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from image_processing.image_proc_rgb import camera_image
from nets.pilotNet_5_layer_128_rgb import PilotNet


dataset_dir =  '../data/datasets/driving_dataset'
model_dir = '../data/models/nvidia/model_test.ckpt'
steer_image = '../data/.logo/steering_wheel_image.jpg'
log_dir = './logs'

clear_log =  False
num_epochs = 30
batch_size = 128; # 128
learning_rate = 1e-4
L2NormConst = 1e-3;

def train():
    """Train PilotNet model"""

    # delete old logs
    if clear_log:
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)

    with tf.Graph().as_default():
        # construct model
        model = PilotNet()

        # images of the road ahead and steering angles in random order
        # dataset = camera_image(dataset_dir)

        train_vars = tf.trainable_variables()
        # define loss
        loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.steering))) \
               + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


        saver = tf.train.Saver(tf.all_variables())

        tf.summary.scalar("loss", loss)
        merged_summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)

            # logs to Tensorboard
            summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
            save_model_path = log_dir + "/checkpoint/"

            print("--> tensorboard --logdir={} " \
                  "\nThen open http://0.0.0.0:6006/ into your web browser".format(log_dir))

            for epoch in range(num_epochs):

                # shuffle list of images at each epoch 
                dataset = camera_image(dataset_dir)
                num_batches = int(dataset.num_images / batch_size)

                for batch in range(num_batches):

                    imgs, angles = dataset.load_train_batch(batch_size)

                    sess.run(
                        [loss, optimizer],
                        feed_dict={
                            model.image_input: imgs,
                            model.y_: angles,
                            model.keep_prob: 0.8
                        }
                    )

                    if batch % 10 == 0:
                        imgs, angles = dataset.load_val_batch(batch_size)
                        loss_value = sess.run(
                            loss,
                            feed_dict={
                                model.image_input: imgs,
                                model.y_: angles,
                                model.keep_prob: 1.0
                            }
                        )
                        
                        print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + batch, loss_value))

                    # write logs
                    summary = merged_summary_op.eval(
                        feed_dict={
                            model.image_input: imgs,
                            model.y_: angles,
                            model.keep_prob: 1.0
                        }
                    )
                    summary_writer.add_summary(summary, epoch * num_batches + batch)
                    summary_writer.flush()

                    # Save the model checkpoint
                    if batch % batch_size == 0:
                        if not os.path.exists(save_model_path):
                            os.makedirs(save_model_path)
                        checkpoint_path = os.path.join(save_model_path, "model.ckpt")
                        filename = saver.save(sess, checkpoint_path)

                print("Model saved in file: %s" % filename)


train();