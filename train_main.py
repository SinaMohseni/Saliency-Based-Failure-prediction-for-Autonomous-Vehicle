#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import datetime
import time as seconds
import numpy as np
import tensorflow as tf
from img_prc.dataset_synth import training_set
from tensorflow.core.protobuf import saver_pb2
from networks.pilotNet_5_layer_128_rgb import PilotNet



dataset_dir =  ['/media/jyadawa/SSD3_3T/sina/datasets/LakeSideTrack/']  # LakeSideTrack
                # '/media/jyadawa/SSD3_3T/sina/datasets/JungleTrack/']    # 320*160 images
training_dir = '/media/jyadawa/SSD3_3T/sina/results/pilotnet/regression/'

state = {}
state['tensorboard'] = False;

training_dir  = training_dir + "/PilotNet_"+ str(datetime.datetime.now()).split(".")[0].replace(" ","_"); 

clear_log =  False
num_epochs = 30
batch_size = 128; 
learning_rate = 1e-5;
L2NormConst = 1e-3;


def train(training_dir,dataset_dir):
    """Train PilotNet model"""

    with tf.Graph().as_default():
        # construct model
        model = PilotNet()

        train_vars = tf.trainable_variables()
        # define loss
        loss = tf.reduce_mean(tf.square(tf.subtract(model.target, model.steering_output))) \
               + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        saver = tf.train.Saver(tf.all_variables())

        tf.summary.scalar("loss", loss)
        merged_summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()


        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            
            # if clear_log:
            #     if tf.gfile.Exists(log_dir):
            #         tf.gfile.DeleteRecursively(log_dir)
            #     tf.gfile.MakeDirs(log_dir)

            # if state['tensorboard'] tensorboard_log():
            save_path = training_dir;
            model_path = save_path + "/checkpoint_models/"
            log_dir = save_path + "/training_log/";

            tf.gfile.MakeDirs(log_dir)
            tf.gfile.MakeDirs(model_path)

            # logs to Tensorboard
            summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
            print("--> tensorboard --logdir={} " \
                  "\nThen open http://0.0.0.0:6006/ into your web browser".format(log_dir))

            
            with open(os.path.join(log_dir + 'training_results.csv'), 'a') as f:
                f.write('epoch,time(s),train_loss,test_loss\n')

            print('Beginning Training\n')
            last_best_lost = 100;
            last_best_epoch = 0;
            last10_test_lost = 100;

            print ("____loading____")
            dataset = training_set(dataset_dir, test_size = 0.2)
            print ("loading____done")

            for epoch in range(num_epochs):
                state['epoch'] = epoch;
                begin_epoch = seconds.time(); # datetime.datetime.utcnow()/1000;

                dataset.shuffle_all(); 

                # --------------- Train ------------------
                num_batches = int(dataset.num_train_images / batch_size)  # int(dataset.num_images / batch_size)
                tot_loss = 0;

                for batch in range(num_batches):

                    imgs, angles = dataset.load_train_batch(batch_size)
                    angles = np.expand_dims(angles, axis=1) 
                    sess.run(
                        optimizer,
                        feed_dict={
                            model.input_img: imgs,
                            model.target: angles,
                            model.keep_prob: 0.9
                        }
                    )
                    # if batch % 10: 
                    loss_value = sess.run(
                        loss,
                        feed_dict={
                            model.input_img: imgs,
                            model.target: angles,
                            model.keep_prob: 1.0
                        }
                    )
                    
                    tot_loss += float(loss_value);

                state['train_loss'] = tot_loss / num_batches; #(dataset.num_train_images/10);

                # --------------- Test ------------------
                # validate and save at each epoch
                num_batches = int(dataset.num_val_images / batch_size)  # int(dataset.num_images / batch_size)
                tot_loss = 0;
                for batch in range(num_batches):

                    imgs, angles = dataset.load_val_batch(batch_size)
                    angles = np.expand_dims(angles, axis=1) 

                    loss_value = sess.run(
                        loss,
                        feed_dict={
                            model.input_img: imgs,
                            model.target: angles,
                            model.keep_prob: 1.0
                        }
                    )

                    tot_loss += float(loss_value);

                state['test_loss'] = tot_loss / num_batches; # dataset.num_val_images;
                    # state['test_accuracy'] = correct / len(test_loader.dataset)

                print ('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f}'.format( #  | Test Error {4:.2f}
                    (epoch + 1),
                    int(seconds.time() - begin_epoch),
                    state['train_loss'],
                    state['test_loss']) # , 100 - 100. * state['test_accuracy']
                )

                # Save Model 
                if state['test_loss'] < last_best_lost:

                    # Save the model checkpoint at each epoch
                    this_checkpoint = os.path.join(model_path, "model_"+str(epoch)+".ckpt")
                    filename = saver.save(sess, this_checkpoint)

                    # Let us not waste space and delete the previous model
                    prev_path = os.path.join(model_path, "model_"+str(last_best_epoch)+".ckpt.data-00000-of-00001")  # epoch - 1
                    if os.path.exists(prev_path): os.remove(prev_path) 
                    prev_path = os.path.join(model_path, "model_"+str(last_best_epoch)+".ckpt.index")
                    if os.path.exists(prev_path): os.remove(prev_path) 
                    prev_path = os.path.join(model_path, "model_"+str(last_best_epoch)+".ckpt.meta")
                    if os.path.exists(prev_path): os.remove(prev_path) 
                    last_best_epoch = epoch;
                    last_best_lost = state['test_loss'];              

                    # Show results
                    with open(os.path.join(log_dir + 'training_results.csv'), 'a') as f:
                        f.write('%03d,%05d,%0.6f,%0.5f\n' % ( # ,%0.2f
                            (epoch + 1),
                            seconds.time() - begin_epoch,
                            state['train_loss'],
                            state['test_loss'],
                            # 100 - 100. * state['test_accuracy'],
                        ));

                
                if epoch+1 % 10 == 0:
                    if state['test_loss'] > last10_test_lost:
                        print ("Early stop: No learning in the past 10 epochs")
                        break;
                    last10_test_lost = state['test_loss']



                
               



train(training_dir,dataset_dir);

print("Training Report in: ", training_dir)