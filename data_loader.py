# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pickle
import numpy as np
import random

'''
summary
1) 작성자 : 박민지
2) 작성일 : 02-12
3) 함수설명 : (1) 클래스마다 가진 이미지들을 랜덤하게 중복없이 차례대로 가져오는 함수
4) 파라미터 : (1) label_list : 이미지들의 라벨을 담는 리스트 
             (2) img_list : 이미지들을 담는 리스트
             (3) fork_epoch : fork 했을 때 이전 epoch들을 가져옴 string 타입
             (4) nb_epoch :  epoch 이 for문에서 돌 때마다 가져옴 int 타입
5) return img_list와 label_list             
'''

def train_data_balancing(data_path, img_size,fork_epoch, nb_epoch):
    # nb_epoch은 이전까지 돌아간 epoch 수
    label_list = []
    img_list = []
    label_idx = 0 # 라벨을 지정해주기 위한 증가변수 (int)
    fork_epoch = int(fork_epoch)

    if fork_epoch == 0:
        fork_epoch = 0
    else:
        fork_epoch +=1
    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        new_epoch = fork_epoch+nb_epoch # 현재 세션 epoch +  fork 세션 epoch
        #print("seed "+str(new_epoch//len(files)))
        np.random.seed(new_epoch//len(files)) #seed 설정 : 돌아간 epoch을 이미지 파일개수로 나누어 몫을 구하면 이미지들이 다 순환되기전까지 seed가 같아짐
        x = np.arange(len(files)) # 리스트를 이미지 파일 개수만큼 생성
        np.random.shuffle(x) # 순서를 바꿔줌 
        filenum = new_epoch % len(files) #균일하게 차례대로 들어가기 위해서는 나머지를 구하면 0 1 2 3 4 5  이렇게 들어감 
        #print("shuffle:"+str(x))
        ''' 이미지 읽어오는 과정'''
        #print("label_idx"+str(label_idx)+"filenum : "+str(filenum))
        filename = files[x[filenum]]  # 선별된 이미지 이름이 들어가도록 x[] 는 shuffle 된 리스트 그 안에 0 1 2 3 4 5 순환되게 들어감 
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
