# -*- coding: utf_8 -*-
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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from data_loader import train_data_loader,train_data_balancing
from keras.applications.densenet import *
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.utils.training_utils import multi_gpu_model
from keras.applications.nasnet import *
import gc
#import autokeras as ak


np.set_printoptions(threshold=np.nan)
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
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm



# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.layers[0].input, outputs=model.layers[-1].output)
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      dtype='float32',
                                      featurewise_std_normalization=True,
                                      featurewise_center=True)

    test_datagen.mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3))
    test_datagen.std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3))

    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs

def balancing_process(train_dataset_path,input_shape, num_classes,nb_epoch):
    img_list = []
    label_list = []
    img_list, label_list = train_data_balancing(train_dataset_path, input_shape[:2], num_classes,nb_epoch)  # nb_epoch은 0~1382개 뽑히는 리스트가 총 몇 번 iteration 하고 싶은지
    # print("list"+str(1)+" label : "+str(label_list[1])+", img : "+str(img_list[1])) 뽑힌 리스트의 내용 확인하는 출력문구

    x_train = np.asarray(img_list, dtype=np.float32)  # (1383, 224, 224, 3)
    labels = np.asarray(label_list)  # (1383,)
    y_train = keras.utils.to_categorical(labels, num_classes=num_classes)  # (1383, 1383)

    return x_train, y_train


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    # epochs가 없으면 fork시 버그 걸려서 넣어둠
    args.add_argument('--epochs', type=int, default=10000)
    args.add_argument('--epoch', type=int, default=10000)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)
    args.add_argument('--lr', type=float, default=0.001)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape
    lr = config.lr
    """ Model """
    model1 = VGG16(input_shape=input_shape, weights=None, include_top = True, classes = num_classes)
    model = DenseNet201(input_shape=input_shape, weights=None, include_top = True, classes = num_classes)
    #model = NASNetMobile(input_shape=input_shape, weights=None, include_top = True, classes = num_classes)
    model = multi_gpu_model(model, gpus=2)
    model.summary()


    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        #nsml.load(checkpoint=151, session='team_33/ir_ph2/141')           # load시 수정 필수!


        """ Initiate RMSprop optimizer """
        #opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)
        opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True, decay= 1e-5)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        train_dataset_path = DATASET_PATH + '/train/train_data'

        # type 그대로 python에서 만들어지는 모든 것들예) array([ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318,  0.4236548 ,]
        # with 문 자동으로 파일 close 해줌

        # img_list,label_list = train_data_balancing(train_dataset_path, input_shape[:2],  num_classes, nb_epoch) #nb_epoch은 0~1382개 뽑히는 리스트가 총 몇 번 iteration 하고 싶은지
        # print("list"+str(1)+" label : "+str(label_list[1])+", img : "+str(img_list[1])) 뽑힌 리스트의 내용 확인하는 출력문구
        # x_train = np.asarray(img_list) #(1383, 224, 224, 3)
        # labels = np.asarray(label_list) #(1383,)
        # y_train = keras.utils.to_categorical(labels, num_classes=num_classes)  #(1383, 1383)
        # x_train = x_train.astype('float32')

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            featurewise_center=True,
            featurewise_std_normalization=True
        )

        # mean RGB: [144.62598745, 132.1989693, 119.10957842]
        # std RGB: [5.71350834, 7.67297079, 8.68071288]
        train_datagen.mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1,1,3))
        train_datagen.std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3))


        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3, verbose=1)


        # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
        #                    steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

        # here's a more "manual" example

        t0 = time.time()
        for e in range(nb_epoch):
            t1 = time.time()
            print('Epochs : ', e)
            #batches = 0
            '''epoch에 맞게 x_rain,y_train가져오기 '''
            x_train, y_train = balancing_process(train_dataset_path, input_shape, num_classes, e)
            #model.evaluate(x_train, y_train, batch_size=batch_size,verbose=1)
            #print(x_train.shape)

            train_generator = train_datagen.flow(
                x_train, y_train,
                batch_size=batch_size,
                shuffle=True,
            )

            STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=e,
                                      epochs=e + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True,
                                      )

            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            #val_loss, val_acc = res.history['val_loss'][0], res.history['val_acc'][0]
            nsml.report(summary=True, epoch=e, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)#, val_loss=val_loss, val_acc=val_acc)
            if (e+1) % 40 == 0:
                nsml.save(e)
                print('checkpoint name : ' + str(e))
            # 메모리 해제
            del x_train
            del y_train
            gc.collect()
        print('Total training time : %.1f' % (time.time() - t0))


        '''
            for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=batch_size):

                #print(x_batch.shape)
                res = model.fit(x_batch,
                              y_batch,
                              callbacks=[reduce_lr],
                              verbose=0,
                              shuffle=True
                              )

            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            #val_loss, val_acc = res.history['val_loss'][0], res.history['val_acc'][0]
            nsml.report(summary=True, epoch=e, epoch_total=nb_epoch, loss=train_loss, acc=train_acc) #, val_loss=val_loss, val_acc=val_acc)
            if (e-1) % 5 == 0:
                nsml.save(e)
                print('checkpoint name : ' + str(e))
            batches += 1
            if batches >= len(x_train) / batch_size:
                # we need to break the loop by hand becauseba
                # the generator loops indefinitely
                break
        print('Total training time : %.1f' % (time.time() - t0))
        '''
       





