# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pickle

def train_data_balancing(data_path, img_size,fork_epoch, nb_epoch):
    # nb_epoch은 이전까지 돌아간 epoch 수
    label_list = []
    img_list = []
    label_idx = 0
    fork_epoch = int(fork_epoch)
    if fork_epoch == 0:
        fork_epoch = 0
    else:
        fork_epoch +=1
    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        filenum = (fork_epoch+nb_epoch) % len(files) #checkpoint 다음이어서 +1

        ''' 이미지 읽어오는 과정'''
        #print("fork_epoch"+fork_epoch+"filenum : "+str(filenum))
        filename = files[filenum]  # 선별된 이미지 이름이 들어가도록 ex) 1. class_list에서 클래스에 맞는 filenum을 찾고 2. filenum을 files의 인덱스로
        img_path = os.path.join(root, filename)
        try:
            img = cv2.imread(img_path, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
        except:
            continue
        label_list.append(label_idx)
        img_list.append(img)
        label_idx += 1
    # print("label_len : " + str(len(label_list)))
    # print( "img_len : " + str(len(img_list)))
    return img_list, label_list
def train_data_loader(data_path, img_size, output_path):
    label_list = []
    img_list = []
    label_idx = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
            except:
                continue
            label_list.append(label_idx)
            img_list.append(img)
        label_idx += 1

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_list, img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_list, label_f)


# nsml test_data_loader
def test_data_loader(data_path):
    data_path = os.path.join(data_path, 'test', 'test_data')

    # return full path
    queries_path = [os.path.join(data_path, 'query', path) for path in os.listdir(os.path.join(data_path, 'query'))]
    references_path = [os.path.join(data_path, 'reference', path) for path in
                       os.listdir(os.path.join(data_path, 'reference'))]

    return queries_path, references_path


if __name__ == '__main__':
    query, refer = test_data_loader('./')
    print(query)
    print(refer)
