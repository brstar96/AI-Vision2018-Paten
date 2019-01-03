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

# NSML 클라우드로부터 ir_ph1데이터셋 주소를 불러옴
from nsml import DATASET_PATH

import keras
from keras.models import Sequential #NN을 초기화할때 필요한 모듈
from keras.layers import Dense, Dropout, Flatten, Activation #Dense : NN을 구성하는 Layer를 생성할때 필요한 모듈
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from data_loader import train_data_loader


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        #data_loader.py의 test_data_loader 함수에 DATASET_PATH를 인자로 넘겨서 queries_path, references_path 반환받으면 될듯

        #224*224해상도의 RGB 이미지로 전처리수행
        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        query_img = query_img.astype('float32')
        query_img /= 255
        reference_img = reference_img.astype('float32')
        reference_img /= 255

        #model로부터 특정
        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-2].output])

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
        #유클리드 공간에서 두 벡터를 내적한 것(sim_matrix)을 query image와 reference img 간의 유사도로 삼음.
        sim_matrix = np.dot(query_vecs, reference_vecs.T)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # nsml.bind : NSML에서 내부적으로 사용하는 함수 또는 변수들을 하나로 묶어주는 함수
    # (https://n-clair.github.io/vision-docs/_build/html/ko_KR/contents/nsml_library/nsml_bind.html 참고)
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
        img = cv2.imread(img_path, 1)  # 이미지 파일을 알파 없는 BGR로 읽어옴 (https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR->RGB로 채널순서변경
        img = cv2.resize(img, img_size)  # 224*224 이미지해상도변경
        query_img.append(img)

    for img_path in db: #상동
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=128)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
    num_classes = 1000
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    model = Sequential() #Neural Net 초기화
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()

    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)

        #학습프로세스 설정
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl'] #train에 사용할 image pickle과 label pickle의 출력경로 설정
        train_dataset_path = DATASET_PATH + '/train/train_data' #학습 인풋데이터 경로설정
        # train_dataset_path = '로컬경로'  # Local에서 테스트할경우 사용하는 코드

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
                       output_path=output_path)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
            train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)

        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f) #train에 사용할 refenence img의 pickle을 load(data_loader.py의 train_data_loader참고)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f) #마찬가지로 label정보만 담긴 pickle을 load

        x_train = np.asarray(img_list) #x_train = reference imgs
        labels = np.asarray(label_list)
        y_train = keras.utils.to_categorical(labels, num_classes=num_classes) #y_train = ref img labels
        x_train = x_train.astype('float32')
        x_train /= 255 #0.0~1.0 실수범위로 데이터를 축소하기 위해 255 컬러범위로 나눔
        print(len(labels), 'train samples')

        #이 부분은 NSML가이드 다시 확인해볼것
        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        for epoch in range(nb_epoch): #default nb_epoch : 5, 다른 파라미터는 args참고
            res = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            initial_epoch=epoch,
                            epochs=epoch + 1,
                            callbacks=[reduce_lr],
                            verbose=1,
                            shuffle=True)
            print(res.history)
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
