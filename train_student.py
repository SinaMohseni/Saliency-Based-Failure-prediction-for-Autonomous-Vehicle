#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import scipy.misc
import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from image_processing.image_proc_gray import camera_image
from nets.pilotNet_5_student_gray import PilotNet


dataset_dir =  '../data/saliency_dataset_3'  
model_dir = './logs/checkpoint_5_layer_lrg4/checkpoint/model.ckpt' 
log_dir = './logs/student_8'
clear_log =  False
num_epochs = 20;
batch_size = 128;
learning_rate = 1e-5
L2NormConst = 1e-3;


def train():

    # delete old logss
    if clear_log:
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
   

    with tf.Graph().as_default():
        
        model = PilotNet();

        saver = tf.train.Saver()

        dataset = camera_image(dataset_dir)

        train_vars = tf.trainable_variables()
        

        # define loss
        loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.steering))) \
               + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) # var_list=second_train_vars

        

        saver = tf.train.Saver(tf.all_variables())
        
        # summary for tensorboard
        tf.summary.scalar("loss", loss)
        
        merged_summary_op = tf.summary.merge_all()
        
        init = tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)

            # restore a pretrained model 
            saver.restore(sess, model_dir)
            

            summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
            save_model_path = log_dir + "/checkpoint/"

            print("Run the command line:\n", "--> tensorboard --logdir={} " \
                  "\nThen open http://0.0.0.0:6006/ into your web browser".format(log_dir))

            for epoch in range(num_epochs):

                num_batches = int(dataset.num_images / batch_size)
                for batch in range(num_batches):
                    imgs, angles = dataset.load_train_batch(batch_size)
                    # imgs = numpy.expand_dims(imgs, axis=-1);

                    # run backprop and calculate loss
                    sess.run(
                        [loss, optimizer],
                        feed_dict={
                            model.image_input: imgs,
                            model.y_: angles,
                            model.keep_prob: 0.6
                        }
                    )

                    if batch % 10 == 0:
                        imgs, angles = dataset.load_val_batch(batch_size)
                        # imgs = numpy.expand_dims(imgs, axis=-1); 

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

                    # Save the model checkpoint periodically.
                    if batch % batch_size == 0:
                        if not os.path.exists(save_model_path):
                            os.makedirs(save_model_path)
                        checkpoint_path = os.path.join(save_model_path, "model_2.ckpt")
                        filename = saver.save(sess, checkpoint_path)

                print("Model saved in file: %s" % filename)

train()