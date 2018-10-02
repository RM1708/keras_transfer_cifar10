'''
https://alexisbcook.github.io/2017/using-transfer-learning-to-classify-images-with-keras/
'''
# coding: utf-8

# # Extract Bottleneck Features for Train Set

# In[ ]:


#import keras
from keras.datasets import cifar10
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import scipy
#from scipy import misc
import os

'''
Have had to reduce the Training and Test data size on account of Memory Error in
the function np.array() 

'''
REDUCED_TRAINING_SAMPLE_COUNT = 5000
REDUCED_TEST_SAMPLE_COUNT = REDUCED_TRAINING_SAMPLE_COUNT//5

# load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:REDUCED_TRAINING_SAMPLE_COUNT]
y_train = y_train[:REDUCED_TRAINING_SAMPLE_COUNT]
x_test = x_test[:REDUCED_TEST_SAMPLE_COUNT]
y_test = y_test[:REDUCED_TEST_SAMPLE_COUNT]

y_train = np.squeeze(y_train)
print('data loaded')


# In[ ]:



# load inceptionV3 model + 
#remove final classification layers by setting include_top to False
model = InceptionV3(weights='imagenet', \
                    include_top=False, \
                    input_shape=(139, 139, 3))
print('model loaded')

# obtain bottleneck features_train
if os.path.exists('inception_features_train.npz'):
    print('bottleneck features_train detected (train). Loading features_train ...')
    features_train = np.load('inception_features_train.npz')['features']
else:
    print('bottleneck features_train file not detected (train)')
    print('calculating now ...')
    # pre-process the train data
    tmp = []
    for i in range(len(x_train)//1):
        if(0 == i%100):
            print(" ", i, end='')
        tmp.append(scipy.misc.imresize(x_train[i], \
                                                    (139, 139, 3)))
    big_x_train = (np.array(tmp)).astype('float32')
#        big_x_train = np.array(scipy.misc.imresize(x_train[i], \
#                                                    (139, 139, 3)).astype('float32')) 
#    big_x_train = np.array([scipy.misc.imresize(x_train[i], (139, 139, 3)) 
#                            for i in range(0, len(x_train))]).astype('float32')
    inception_input_train = preprocess_input(big_x_train)
    print('train data preprocessed')
    # extract, process, and save bottleneck features_train
    features_train = model.predict(inception_input_train)
    features_train = np.squeeze(features_train)
    np.savez('inception_features_train', features=features_train)
    print('bottleneck features_train saved')


# # Plot t-SNE Embedding of Bottleneck Features for Train Set

# In[ ]:


from tsne import bh_sne
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# reshape bottleneck features_train + reduce dimensionality with t-SNE
if os.path.exists('tsne_features_train.npz'):
    print('tsne features detected (test). Loading t-SNE features ...')
    tsne_features_train = np.load('tsne_features_train.npz')['tsne_features']
else:
    print('tsne features not detected (test)')
    print('calculating now ...')
    tsne_features_train = \
                bh_sne(features_train.reshape([features_train.shape[0], \
                                             np.prod(features_train.shape[1:])]).astype('float64')[:25000])
    np.savez('tsne_features_train', tsne_features=tsne_features_train)
    print('tsne features obtained')

# plot the features
plt.figure(figsize=(20,20))
plt.scatter(tsne_features_train[:,0], \
            tsne_features_train[:,1], \
            c=plt.cm.jet(y_train/10), \
            s=10, \
            edgecolors='none')
plt.show()


# # Extract Bottleneck Features for Test Set

# In[ ]:


# obtain bottleneck features (test)
if os.path.exists('inception_features_test.npz'):
    print('bottleneck features_test detected. Loading features_test ...')
    features_test = np.load('inception_features_test.npz')['features_test']
else:
    print('bottleneck features_test file not detected')
    print('calculating now ...')
    # pre-process the test data
    big_x_test = np.array([scipy.misc.imresize(x_test[i], (139, 139, 3)) 
                       for i in range(0, len(x_test)//1)]).astype('float32')
    inception_input_test = preprocess_input(big_x_test)
    # extract, process, and save bottleneck features (test)
    features_test = model.predict(inception_input_test)
    features_test = np.squeeze(features_test)
    np.savez('inception_features_test', features_test=features_test)
    print('bottleneck features_test saved')


# # Train a Shallow NN

# In[ ]:


from keras.utils import np_utils

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# In[ ]:


from keras.callbacks import ModelCheckpoint   
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D
#/home/rm/anaconda3/envs/tensorflow/lib/python3.6/site-packages/keras/engine/sequential.py
model = Sequential()
model.add(Conv2D(filters=100, \
                 kernel_size=2, \
                 input_shape=features_train.shape[1:]))
model.add(Dropout(0.4))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(10, \
                activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', \
              optimizer='rmsprop', \
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.best.hdf5', 
                               verbose=1, save_best_only=True)
model.fit(features_train, y_train, batch_size=50, epochs=50,
          validation_split=0.2, callbacks=[checkpointer],
          verbose=2, shuffle=True)


# In[ ]:


# load the weights that yielded the best validation accuracy
model.load_weights('model.best.hdf5')

# evaluate test accuracy
score = model.evaluate(features_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('\nTest accuracy: %.4f%%' % accuracy)

