import ants
import antspynet
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import glob

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/reproduce-chexnet/'
scripts_directory = base_directory + 'antsxnet/scripts/'
data_directory = base_directory + "data/"

population_prior = ants.image_read(antspynet.get_antsxnet_data("xrayLungPriors"))
population_prior = (ants.slice_image(population_prior, axis=2, idx=0, collapse_strategy=1) +
                    ants.slice_image(population_prior, axis=2, idx=1, collapse_strategy=1))
image_size=(224, 224)
population_prior = ants.resample_image(population_prior, image_size, use_voxels=True)

################################################
#
#  Load the data
#
################################################

train_images_file = base_directory + "antsxnet/train_val_list.txt"
with open(train_images_file) as f:
    train_images_list = f.readlines()
f.close()
train_images_list = [x.strip() for x in train_images_list]

demo_file = base_directory + "antsxnet/Data_Entry_2017_v2020.csv"
demo = pd.read_csv(demo_file)

def unique(list1):
    unique_list = [] 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return sorted(unique_list)

unique_labels = demo['Finding Labels'].unique()
unique_labels_unroll = []
for i in range(len(unique_labels)):
    label = unique_labels[i]
    labels = label.split('|')
    for j in range(len(labels)):
        unique_labels_unroll.append(labels[j])

unique_labels = unique(unique_labels_unroll)

training_demo_file = data_directory + "training_demo.npy"
training_demo = None
if os.path.exists(training_demo_file):
    training_demo = np.load(training_demo_file)
else:
    training_demo = np.zeros((len(train_images_list), len(unique_labels)))
    for i in tqdm(range(len(train_images_list))):
        image_filename = train_images_list[i]
        row = demo.loc[demo['Image Index'] == image_filename]
        findings = row['Finding Labels'].str.cat().split("|")
        for j in range(len(findings)):
            training_demo[i, unique_labels.index(findings[j])] = 1.0
    np.save(training_demo_file, training_demo)        

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = len(unique_labels)
number_of_channels = 3

model = tf.keras.applications.DenseNet121(include_top=True, 
                                          weights=None, 
                                          input_tensor=None, 
                                          input_shape=None, 
                                          pooling=None, 
                                          classes=number_of_classification_labels, 
                                          classifier_activation='sigmoid')

weights_filename = scripts_directory + "xray_classification_with_spatial_priors.h5"
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

###
#
# Set up the training generator
#

random.seed(1234)

random.shuffle(train_images_list)
validation_split = int(0.2 * len(train_images_list))

generator = batch_generator(batch_size=32,
                            image_files=train_images_list[validation_split:],
                            demo=training_demo[validation_split:],
                            image_size=image_size,
                            population_prior=population_prior,
                            do_augmentation=False)

val_generator = batch_generator(batch_size=8,
                            image_files=train_images_list[:validation_split],
                            demo=training_demo[:validation_split],
                            image_size=image_size,
                            population_prior=population_prior,
                            do_augmentation=False)

track = model.fit(x=generator, epochs=10000, verbose=1, steps_per_epoch=100,
                  validation_data=val_generator, validation_steps=1,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='val_loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
#       keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.99999999999999999999999,
#          verbose=1, patience=10, mode='auto')
#       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001,
#          patience=20)
       ]
   )

model.save_weights(weights_filename)


