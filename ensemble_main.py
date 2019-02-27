# Dummy session generation code for ensemble
# This code is written for query expansion with ensemble.

# 1. 1000클래스 Imagenet으로 프리트레인된 DenseNet201 40, 60, 70 Epoch 모델 3개를 nsml.load로 로드합니다.
# 2. 모델 로딩 후 마지막 feature 레이어 3개를 concat합니다.
# 3. 새로운 모델을 nsml.bind로 바인드하고 Data augmentation 과정을 거친 후 Query expansion을 수행합니다.
# 4. 쿼리 이미지와 레퍼런스 이미지 간의 코싸인 유사도를 비교합니다.

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
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Average
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
from keras.applications.nasnet import *
from keras.applications.densenet import *
from data_loader import train_data_loader,train_data_balancing
import gc
#from DenseNet import densenet169

num_classes = 1383
input_shape = (224, 224, 3)  # input image shape

basemodel1 = DenseNet201(input_shape=input_shape, weights=None, include_top=False, classes=1000)
x = basemodel1.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dense(num_classes, activation='softmax', name='fc1383')(x)
DenseNet201_SkeletonModel_1 = Model(inputs=basemodel1.input, outputs=x, name='DtNet201_40')

basemodel2 = DenseNet201(input_shape=input_shape, weights=None, include_top=False, classes=1000)
y = basemodel2.output
y = GlobalAveragePooling2D(name='avg_pool')(y)
y = Dense(num_classes, activation='softmax', name='fc1383')(y)
DenseNet201_SkeletonModel_2 = Model(inputs=basemodel2.input, outputs=y, name='DtNet201_60')

basemodel3 = DenseNet201(input_shape=input_shape, weights=None, include_top=False, classes=1000)
z = basemodel3.output
z = GlobalAveragePooling2D(name='avg_pool')(z)
z = Dense(num_classes, activation='softmax', name='fc1383')(z)
DenseNet201_SkeletonModel_3 = Model(inputs=basemodel3.input, outputs=z, name='DenseNet201_70')

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

        queries, query_vecs, references, reference_vecs = get_feature(basemodel1, basemodel2, basemodel3, queries, db)

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
def get_feature(model1, model2, model3, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3)) / 255.0
    std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3)) / 255.0

    # Create Intermeriate Layer model for Inference
    intermediate_layer_model_from_DenseNet201_40 = Model(inputs=model1.layers[0].input, outputs=model1.layers[-1].output)
    intermediate_layer_model_from_DenseNet201_60 = Model(inputs=model2.layers[0].input, outputs=model2.layers[-1].output)
    intermediate_layer_model_from_DenseNet201_70 = Model(inputs=model3.layers[0].input, outputs=model3.layers[-1].output)

    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32',
                                      rotation_range=180,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      featurewise_center=True,
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

    query_vecs_from_DenseNet201_40 = intermediate_layer_model_from_DenseNet201_40.predict_generator(query_generator, steps=len(query_generator), verbose=1)
    query_vecs_from_DenseNet201_60 = intermediate_layer_model_from_DenseNet201_60.predict_generator(query_generator, steps=len(query_generator), verbose=1)
    query_vecs_from_DenseNet201_70 = intermediate_layer_model_from_DenseNet201_70.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    query_vecs_from_DenseNets = []
    query_vecs_from_DenseNets.append(query_vecs_from_DenseNet201_40)
    query_vecs_from_DenseNets.append(query_vecs_from_DenseNet201_60)
    query_vecs_from_DenseNets.append(query_vecs_from_DenseNet201_70)
    AveragedQueryVecs = Average()(query_vecs_from_DenseNets)


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

    reference_vecs_from_DenseNet201_40 = intermediate_layer_model_from_DenseNet201_40.predict_generator(reference_generator, steps=len(reference_generator), verbose=1)
    reference_vecs_from_DenseNet201_60 = intermediate_layer_model_from_DenseNet201_60.predict_generator(reference_generator, steps=len(reference_generator), verbose=1)
    reference_vecs_from_DenseNet201_70 = intermediate_layer_model_from_DenseNet201_70.predict_generator(reference_generator, steps=len(reference_generator), verbose=1)

    ref_vecs_from_DenseNets = []
    ref_vecs_from_DenseNets.append(reference_vecs_from_DenseNet201_40)
    ref_vecs_from_DenseNets.append(reference_vecs_from_DenseNet201_60)
    ref_vecs_from_DenseNets.append(reference_vecs_from_DenseNet201_70)
    AveragedRefVecs = Average()(ref_vecs_from_DenseNets)

    return queries, AveragedQueryVecs, db, AveragedRefVecs

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
    args.add_argument('--epochs', type=int, default=1)
    args.add_argument('--epoch', type=int, default=1)
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

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Load Trained models """
        bind_model(model=DenseNet201_SkeletonModel_1)  # weight가 로드된 모델이 생성됨.
        nsml.load(checkpoint = 'DtNet201_40', session='team_33/ir_ph2/607')  # loading trained densenet169
        nsml.save('save1')
        bind_model(model=DenseNet201_SkeletonModel_2)  # weight가 로드된 모델이 생성됨.
        nsml.load(checkpoint = 'DtNet201_60', session='team_33/ir_ph2/607')  # loading trained densenet201
        nsml.save('save2')
        bind_model(model=DenseNet201_SkeletonModel_2)  # weight가 로드된 모델이 생성됨.
        nsml.load(checkpoint='DtNet201_70', session='team_33/ir_ph2/607')  # loading trained densenet201
        nsml.save('save3')
        exit()