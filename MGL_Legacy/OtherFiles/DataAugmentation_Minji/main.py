# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import pickle

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from data_loader import train_data_loader
from keras.preprocessing.image import ImageDataGenerator
import densenet

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        print('model load start!')
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        print('queries')
        print(queries)
        print('references')
        print(references)

        query_img = query_img.astype('float32')
        query_img /= 255

        # 색 반전
        query_img = 1 - query_img

        reference_img = reference_img.astype('float32')
        reference_img /= 255

        # 색 반전
        reference_img = 1 - reference_img

        # 확률 형태로 vector 뽑아냄
        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])

        # get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-2].output])

        print('inference start')

        # inference
        query_vecs = get_feature_layer([query_img, 0])[0]

        # caching db output, db inference
        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            reference_vecs = get_feature_layer([reference_img, 0])[0]
            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        print('sim_matrix')
        print(sim_matrix)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            print('query')
            print(query)
            query = query.split('/')[-1].split('.')[0]
            print('after split query')
            print(query)
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')
        print(list(zip(range(len(retrieval_results)), retrieval_results.items())))

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=32)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
    num_classes = 1000
    input_shape = (224, 224, 3)  # input image shape


    """ Densenet Model """
    model = densenet.DenseNet()
    model.summary()

    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':

        # nsml.load(checkpoint='submit2', session='team_33/ir_ph1_v2/38')           # load시 수정 필수!

        bTrainmode = True

        """ Initiate RMSprop optimizer """
        # opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        opt = keras.optimizers.Adam(lr = 1e-4)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        #nsml server에서 실행하는냐, 로컬에서 실행하느냐
        if nsml.IS_ON_NSML:
            # Caching file
            #nsml.cache는 preprocess의 과정을 거친 값을 캐싱해 놓음, train_data_lader 는 preprocess의 과정을 거친다. (224,224,3)말고 (224,224)이미지 size
            #train_data_loader는 data_loader.py 안에서 preprocess 시행 결과 : img_list, label_list 생성
            #train_data_lader에서 ouput path에 저장해놓음 img_list는 ./img_list.pkl에 label_list는 ./label_list.pkl에
            nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
                       output_path=output_path)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
            train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)

        #아래 과정에서 읽기 시작함
        #pickle (현재 메모리에 살아있는 ?) 파이썬 객체 자체를 읽고 저장하기 위해
        #type 그대로 python에서 만들어지는 모든 것들예) array([ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318,  0.4236548 ,]
        #with 문 자동으로 파일 close 해줌
        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)

        #numpy 연산 가능하도록 numpy array로 바꾸는 게 핵심
        x_train = np.asarray(img_list)
        labels = np.asarray(label_list)
        #목적 : categorical_crossentropy 사용을 위해, integer 를 binary class matrix로 반환
        #shape(7,7) num class가 7인경우
        #naver 예제의 keras to categorical 거치면 (7064,1000) shape의 형태
        '''
        [[1. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 0. 1.]]
 '''
        y_train = keras.utils.to_categorical(labels, num_classes=num_classes)
        x_train = x_train.astype('float32')


        x_train /= 255

        x_train = 1 - x_train
        print(len(labels), 'train samples')

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)


        """ Training loop """
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)

        datagen.fit(x_train)
        #fig_generator에서 epoch 루프 수행
        res = model.fit_generator(
          datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train)/32,
          epochs= nb_epoch )

        for epoch in range(nb_epoch):
            print('Epoch', epoch + 1)
            batches = 0
            for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
                res = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                initial_epoch=epoch,
                                epochs=epoch + 1,
                                callbacks=[reduce_lr],
                                verbose=1,
                                shuffle=True)
                batches += 1
                if batches >= len(x_train) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
                print(res.history)
                train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
                nsml.report(summary=True,  epoch=epoch, epoch_total=nb_epoch,loss=train_loss, acc=train_acc)
                if epoch % 10 == 0:
                    check = "DN_model_2_aug_"+str(epoch)
                    # check = 'submit2'
                    print('checkpoint name : '+ check)
                    nsml.save(checkpoint=check)
