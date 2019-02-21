# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pickle
import numpy as np
import random

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
        if (len(files) >= 10):
            filenum = (fork_epoch + nb_epoch) % (len(files) - 1)  # checkpoint 다음이어서 +1
            ''' 이미지 읽어오는 과정'''
            #if (label_idx < 10):
                # print("label_idx" + str(label_idx) + "filenum : " + str(filenum))
                #print("files" + str(len(files) - 1))
            new_epoch = fork_epoch+nb_epoch
            #print("seed "+str(new_epoch//len(files)))
            np.random.seed(new_epoch//(len(files)-1))
            x = np.arange(len(files)-1)
            np.random.shuffle(x)
            filenum = new_epoch % (len(files)-1) #균일하게
            #print("shuffle:"+str(x))
            ''' 이미지 읽어오는 과정'''
            #print("label_idx"+str(label_idx)+"filenum : "+str(filenum))
            filename = files[x[filenum]]  # 선별된 이미지 이름이 들어가도록 ex) 1. class_list에서 클래스에 맞는 filenum을 찾고 2. filenum을 files의 인덱스로
            #print("xfilenum"+ str(x[filenum]))
            #print("files" + str(files[x[filenum]]))
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
        else :
            #만약 클래스의 이미지 개수가 10보다 작으면 평소대로 가져오기
            filenum = (fork_epoch+nb_epoch) % len(files) #checkpoint 다음이어서 +1
            ''' 이미지 읽어오는 과정'''
            if (label_idx < 10):
                print("label_idx"+str(label_idx)+"filenum : "+str(filenum))
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
            print("label_idx"+str(label_idx)+"filename"+filename )
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
            except:
                continue
            label_list.append(label_idx)
            img_list.append(img)
        label_idx += 1
    return label_list,img_list
    # write output file for caching
   # with open(output_path[0], 'wb') as img_f:
   #     pickle.dump(img_list, img_f)
    #with open(output_path[1], 'wb') as label_f:
     #   pickle.dump(label_list, label_f)

def val_data_loader(data_path, img_size):
    label_list_val = []
    img_list_val = []
    label_idx = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        if (len(files) >= 10):
            filenum = len(files) - 1  #  클래스의 맨 마지막에 있는 이미지를 가져오고 싶음 6개 있으면 5를 가져와야함
            ''' 이미지 읽어오는 과정'''
            #print("label_idx : "+str(label_idx) + "val_filenum : "+str(filenum))
            filename = files[filenum]  # 선별된 이미지 이름이 들어가도록 ex) 1. class_list에서 클래스에 맞는 filenum을 찾고 2. filenum을 files의 인덱스로
            img_path = os.path.join(root, filename)
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
            except:
                continue
            label_list_val.append(label_idx)
            img_list_val.append(img)
            label_idx += 1

        else:
            # 만약 클래스의 이미지 개수가 10보다 작으면 건너뛰기, label_idx는 증가 [0,2,5,] 10개면 1108개
            label_idx += 1
            continue

    print("label_val_len : " + str(len(label_list_val)))
    print("img_val_len : " + str(len(img_list_val)))
    return img_list_val, label_list_val

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
