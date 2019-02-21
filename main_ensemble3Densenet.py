# -*- coding: utf_8 -*-
# CNN + XGBoost + Softmax 구현을 위한 baselinecode (AvgEnsemble코드에서 fork)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import nsml
import numpy as np
import h5py

from nsml import DATASET_PATH
print(DATASET_PATH)
print (os.getcwd()) #현재 디렉토리의
print (os.path.realpath(__file__))#파일
print (os.path.dirname(os.path.realpath(__file__)) )#파일이 위치한 디렉토리

import keras
from keras.callbacks import History
from keras import Input, layers
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D, Average
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from data_loader import train_data_balancing

# pretrained models from Keras
from keras.applications.inception_v3 import *
from keras.applications.resnet50 import *
from keras.applications.densenet import *

from keras.utils.training_utils import multi_gpu_model
import gc

train_dataset_path = DATASET_PATH + '/train/train_data'
mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3)) / 255.0
std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3)) / 255.0

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

    mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3)) / 255.0
    std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3)) / 255.0

    intermediate_layer_model = Model(inputs=model.layers[0].input, outputs=model.layers[-1].output)
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      dtype='float32',
                                      featurewise_std_normalization=True,
                                      featurewise_center=True)

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

def balancing_process(train_dataset_path,input_shape, fork_epoch ,nb_epoch):
    img_list, label_list = train_data_balancing(train_dataset_path, input_shape[:2], fork_epoch,nb_epoch)  # nb_epoch은 0~1382개 뽑히는 리스트가 총 몇 번 iteration 하고 싶은지
    # print("list"+str(1)+" label : "+str(label_list[1])+", img : "+str(img_list[1])) 뽑힌 리스트의 내용 확인하는 출력문구

    x_train = np.asarray(img_list, dtype=np.float32)  # (1383, 224, 224, 3)
    labels = np.asarray(label_list)  # (1383,)
    y_train = keras.utils.to_categorical(labels, num_classes=num_classes)  # (1383, 1383)

    return x_train, y_train

def AddFineTuningLayer(basemodel, model_input, modelname):
    # include_top = False이면 FCN같은 마지막 레이어 미포함
    # These classification layer structures are from Keras official github and paper
    if modelname == 'InceptionV3':
        # basemodel.trainable = False
        x = basemodel.output
        # Classification block (codes from 'inception_v3.py' on keras official github)
        # avg_pool (GlobalAveragePooling2 (None, 2048) 0 activation_143[0][0]
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(config.num_classes, activation='softmax', name='fc1383')(x)
        model = Model(inputs = basemodel.input, outputs = x, name = 'InceptionV3')
        # model = Model(inputs=model_input, outputs=x, name='InceptionV3') # 위에 코드 안먹히면 이걸로 다시 테스트 해볼것.
        bind_model(model)
        for i in range(0, 3):
            print('')
        print('=================== ' + modelname + ' has been successfully Modfied! ===================')
        model.summary()
        return model
    elif modelname == 'ResNet50':
        # basemodel.trainable = False
        x = basemodel.output
        # avg_pool (GlobalAveragePooling2 (None, 2048) 0 activation_143[0][0]
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(config.num_classes, activation='softmax', name='fc1383')(x)
        model = Model(inputs = basemodel.input, outputs = x, name = 'ResNet50')
        bind_model(model)
        for i in range(0, 3):
            print('')
        print('=================== ' + modelname + ' has been successfully Modfied! ===================')
        model.summary()
        return model
    elif modelname == 'DenseNet169':
        # basemodel.trainable = False
        x = basemodel.output
        # avg_pool (GlobalAveragePooling2 (None, 1920) 0 relu[0][0]
        x = GlobalAveragePooling2D(name='avg_pool')(x)  # same as ResNet50
        x = Dense(config.num_classes, activation='softmax', name='fc1383')(x)  # same as ResNet50
        model = Model(inputs = basemodel.input, outputs = x, name = 'DenseNet169')
        bind_model(model)
        for i in range(0, 3):
            print('')
        print('=================== ' + modelname + ' has been successfully Modfied! ===================')
        model.summary()
        return model
    else:
        NotImplementedError

def model_Fit(model, Modelname):
    t0 = time.time()
    for e in range(nb_epoch):
        t1 = time.time()
        print('')
        print(Modelname + ' Epochs : ', e+1)
        '''epoch에 맞게 x_rain,y_train가져오기 '''
        x_train, y_train = balancing_process(train_dataset_path, input_shape, st_epoch, e)

        # train_datagen.fit(x_train)
        train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=batch_size,
            shuffle=True,
        )

        # 새로 데이터 넣어줄때마다 mean std 설정
        train_datagen.mean = mean
        train_datagen.std = std

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

        if Modelname == 'InceptionV3':
            print('Model generated : ' + Modelname + "_" + str(e))
            if (e) % 20 == 0:
                CkptName = 'IpNet' + '_' + str(e)
                nsml.save(CkptName)
                print('checkpoint name : ' + str(CkptName))
            if e > 60 and e < 100:
                CkptName = 'tIpNet' + '_' + str(e)
                nsml.save(str(CkptName))
                print('checkpoint name : ' + str(CkptName))
        elif Modelname == 'ResNet50':
            print('Model generated : ' + Modelname + "_" + str(e))
            if (e) % 20 == 0:
                CkptName = 'RNNet' + '_' + str(e)
                nsml.save(CkptName)
                print('checkpoint name : ' + str(CkptName))
            if e > 60 and e < 100:
                CkptName = 'tRNNet' + '_' + str(e)
                nsml.save(str(CkptName))
                print('checkpoint name : ' + str(CkptName))
        elif Modelname == 'DenseNet169':
            print('Model generated : ' + Modelname + "_" + str(e))
            if (e) % 20 == 0:
                CkptName = 'DtNet' + '_' + str(e)
                nsml.save(CkptName)
                print('checkpoint name : ' + str(CkptName))
            if e > 60 and e < 100:
                CkptName = 'tDtNet' + '_' + str(e)
                nsml.save(str(CkptName))
                print('checkpoint name : ' + str(CkptName))
        else:
            NotImplementedError

        # 메모리 해제
        del x_train
        del y_train
        gc.collect()
    print('Total training time : %.1f' % (time.time() - t0))
    return model, res

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    # epochs가 없으면 fork시 버그 걸려서 넣어둠
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
    config = args.parse_args()

    # training parameters
    opt = config.opt
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape
    lr = config.lr
    st_epoch = config.iteration  # fork할 때, balancing count 받아오기 위해서 iteration = start epoch
    mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3)) / 255.0
    std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3)) / 255.0
    ModelNames = ['InceptionV3', 'ResNet50', 'DenseNet169']

    """ Load Base Models and Setting Input Shape """
    # weights='imagenet'
    base_model1 = InceptionV3(input_shape = input_shape, weights='imagenet', include_top=False, classes=1000) # base_model1 : <class 'keras.engine.training.Model'>
    base_model2 = ResNet50(input_shape = input_shape, weights='imagenet', include_top=False, classes=1000)
    base_model3 = DenseNet169(input_shape = input_shape, weights='imagenet', include_top=False, classes=1000)

    model_input1 = Input(shape=base_model1.input_shape, name='image_input')
    model_input2 = Input(shape=base_model2.input_shape, name='image_input')
    model_input3 = Input(shape=base_model3.input_shape, name='image_input')

    """ Add Finetuning Layers to pre-trained model and bind to NSML """
    FineTunedInceptionV3 = AddFineTuningLayer(base_model1, model_input1, ModelNames[0]) # FineTunedInceptionV3 : <class 'keras.engine.training.Model'>
    FineTunedResNet50 = AddFineTuningLayer(base_model2, model_input2, ModelNames[1])
    FineTunedDenseNet169 = AddFineTuningLayer(base_model3, model_input3, ModelNames[2])

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        #nsml.load(checkpoint=151, session='team_33/ir_ph2/141')           # load시 수정 필수!

        """ Initiate RMSprop optimizer """
        if (opt == 'rmsprop'):
            opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)
        elif (opt == 'sgd'):
            opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True, decay=1e-6)

        """ Compile 3 Models """
        FineTunedInceptionV3.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # <class 'keras.engine.training.Model'>
        FineTunedResNet50.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        FineTunedDenseNet169.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

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
        train_datagen.mean = mean
        train_datagen.std = std

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3, verbose=1)

        """ Model Fit, Training Loop, Checkpoint save """
        FineTunedInceptionV3 = model_Fit(FineTunedInceptionV3, ModelNames[0])
        del FineTunedInceptionV3
        print(ModelNames[0] + ' has been deleted.')
        FineTunedResNet50 = model_Fit(FineTunedResNet50, ModelNames[1])
        del FineTunedResNet50
        print(ModelNames[1] + ' has been deleted.')
        FineTunedDenseNet169 = model_Fit(FineTunedDenseNet169, ModelNames[2])
        del FineTunedDenseNet169
        print(ModelNames[2] + ' has been deleted.')
        # TrainedModels = [FineTunedInceptionV3[0], FineTunedResNet50[0], FineTunedDenseNet169[0]] # FineTunedInceptionV3[0] : <class 'keras.engine.training.Model'>

    # place ensemble functioncall here