# -*- coding: utf_8 -*-
# 1000클래스 Imagenet으로 프리트레인된 DenseNet169를 파인튜닝하는 코드입니다.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
from keras.applications.nasnet import *
from keras.applications.densenet import *
from data_loader import train_data_loader,train_data_balancing, val_data_loader
import gc
#from DenseNet import densenet169

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norm, where=norm != 0)


# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3)) / 255.0
    std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3)) / 255.0

    intermediate_layer_model = Model(inputs=model.layers[0].input, outputs=model.layers[-1].output)
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32',featurewise_center=True,
            featurewise_std_normalization=True)
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=64,
        class_mode=None,
        shuffle=False
    )

    test_datagen.mean = mean
    test_datagen.std = std

    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)



    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=64,
        class_mode=None,
        shuffle=False
    )

    test_datagen.mean = mean
    test_datagen.std = std

    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs
def balancing_process(train_dataset_path,input_shape, fork_epoch,nb_epoch):
    img_list = []
    label_list = []
    img_list, label_list = train_data_balancing(train_dataset_path, input_shape[:2], fork_epoch,nb_epoch)  # nb_epoch은 0~1382개 뽑히는 리스트가 총 몇 번 iteration 하고 싶은지
    #fork_epoch = int(fork_epoch)+1+nb_epoch
    #print("list"+str(fork_epoch)+" label : "+str(label_list[0])+", img : "+str(img_list[0])) #뽑힌 리스트의 내용 확인하는 출력문구

    x_train = np.asarray(img_list, dtype=np.float32)  # (1383, 224, 224, 3)
    labels = np.asarray(label_list)  # (1383,)
    y_train = keras.utils.to_categorical(labels, num_classes=1383)  # (1383, 1383)

    return x_train, y_train

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)
    args.add_argument('--lr', type=float, default=0.0001)
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--opt', type=str, default='rmsprop')
    args.add_argument('--dropout', type=float, default=0.5)
    args.add_argument('--balepoch', type=str, default=0)
    args.add_argument('--gpus', type=int, default=1)
    config = args.parse_args()

    # training parameters
    opt = config.opt
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape
    lr = config.lr
    dropout = config.dropout
    st_epoch = config.balepoch  # fork할 때, balancing count 받아오기 위해서 iteration = start epoch
    gpus = config.gpus
    mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3)) / 255.0
    std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3)) / 255.0


    """ Model """
    # 'imagenet'
    DenseNet_169 = DenseNet169(input_shape=input_shape, weights='imagenet', include_top=False, classes=1000)

    DenseNet_169.trainable = True
    set_trainable = False

    # 마지막 블록만 학습(파인튜닝)에 사용
    for layer in DenseNet_169.layers:
        if layer.name == 'conv5_block1_0_bn':
            layerNum = layer
            set_trainable = True
            print('3/4 blocks locked, 1/4 unlocked.')
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    for layer in DenseNet_169.layers:
        print(layer, layer.trainable)

    # Create new Sequential Model
    basemodel = Sequential()

    # Add DenseNet169 to basemodel object
    basemodel.add(DenseNet_169)

    # Add New Layers
    basemodel.add(layers.Dense(num_classes, activation = 'relu', name='Fit2NSMLDatasetDim'))
    basemodel.add(layers.GlobalAveragePooling2D(name='GlobalAvgPooling2D'))
    # model = Model(inputs=basemodel.input, outputs='GlobalAvgPooling2D')
    basemodel.summary()

    bind_model(basemodel)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True
        #nsml.load(checkpoint=160, session='team_33/ir_ph2/489')           # load시 수정 필수!
        #nsml.save("489_160")

        if gpus > 1:
            basemodel = multi_gpu_model(basemodel, gpus=2)
            print('uses gpu :' + str(gpus))
        """ Initiate RMSprop optimizer """
        #opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)
        if (opt == 'rmsprop'):
            opt = keras.optimizers.rmsprop(lr=lr, decay=1e-5)

        elif(opt == 'sgd'):
            opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True, decay= 1e-5)

        basemodel.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        print('dataset path', DATASET_PATH)
        train_dataset_path = DATASET_PATH + '/train/train_data'


        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True
        )
        train_datagen.mean = mean
        train_datagen.std = std

        val_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32', featurewise_center=True,
                                         featurewise_std_normalization=True)

        val_datagen.mean = mean
        val_datagen.std = std

        #  Callback
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3, verbose=1)

        img_list_val = []
        label_list_val = []
        img_list_val, label_list_val = val_data_loader(train_dataset_path, input_shape[:2])
        x_test= np.asarray(img_list_val, dtype=np.float32)  #
        labels_test = np.asarray(label_list_val)  #
        y_test = keras.utils.to_categorical(labels_test, num_classes=1383)  # (1383, 1383)


        t0 = time.time()
        for e in range(nb_epoch):
            t1 = time.time()
            print('Epochs : ', e)
            # batches = 0
            '''epoch에 맞게 x_rain,y_train가져오기 '''
            x_train, y_train = balancing_process(train_dataset_path, input_shape, st_epoch, e)
            # model.evaluate(x_train, y_train, batch_size=batch_size,verbose=1)
            # print(x_train.shape)
            train_generator = train_datagen.flow(
                x_train, y_train,
                batch_size=batch_size,
                shuffle=True
            )

            # 새로 데이터 넣어줄때마다 mean std 설정
            train_datagen.mean = mean
            train_datagen.std = std

            validation_generator = val_datagen.flow(
                x_test, y_test,
                batch_size=batch_size,
                shuffle=True
            )

            val_datagen.mean = mean
            val_datagen.std = std

            STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
            res = basemodel.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=e,
                                      epochs=e + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True,
                                      validation_data = validation_generator,
                                      validation_steps = STEP_SIZE_TRAIN
                                      )

            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            val_loss, val_acc = res.history['val_loss'][0], res.history['val_acc'][0]
            nsml.report(summary=True, epoch=e, epoch_total=nb_epoch, loss=train_loss,
                        acc=train_acc   , val_loss=val_loss, val_acc=val_acc)
            if (e + 1) % 40 == 0:
                nsml.save(str(e + 1))
                print('checkpoint name : ' + str(e + 1))
            if e > 60 and e < 90:
                CkptName = 'tIp' + '_' + str(e)
                nsml.save(str(CkptName))
                print('checkpoint name : ' + str(CkptName))


            # 메모리 해제
            del x_train
            del y_train
            gc.collect()
        print('Total training time : %.1f' % (time.time() - t0))
