# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pickle

def train_data_balancing(data_path, img_size,num_classes,nb_epoch):
    class_list = [0] * num_classes  #class마다 이미지가 뽑혔을 때, 증가되는 수를 담기 위한 리스트 [0,0,0,0,0, ... 0] 1383개

    for epoch_num in range(nb_epoch):
        label_list = []
        img_list = []
        label_idx = 0
        for root, dirs, files in os.walk(data_path):
            if not files:
                continue
            filenum = class_list[label_idx]
            print(str(label_idx) + "class, " +str(filenum)+"filenum, "+ str(class_list[label_idx])+"class_list[label_idx]")
            filename = files[filenum] #선별된 이미지 이름이 들어가도록 ex) 1. class_list에서 클래스에 맞는 filenum을 찾고 2. filenum을 files의 인덱스로

            img_path = os.path.join(root, filename)
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
            except:
                continue
            label_list.append(label_idx)
            img_list.append(img)

            filenum +=1 # 매 클래스들마다 재사용됨
            if(filenum==len(files)):
                filenum = 0 # 클래스가 가진 max 이미지 개수에 도달하면 초기화
            class_list[label_idx] = filenum #클래스마다 몇번 째 이미지 인덱스가 뽑혀야하는 지 기록
            label_idx += 1
        print(str(epoch_num)+"nb_epoch, "+"label_len : "+str(len(label_list)))
        print(str(epoch_num)+"nb_epoch, "+"img_len : " + str(len(img_list)))
        print(str(epoch_num)+"nb_epoch, "+"class_list_len : " + str(len(class_list)))
      
    return img_list,label_list

def train_data_loader(data_path, img_size, output_path):
    label_list = []
    img_list = []
    label_idx = 0

    #sum = 0
    #max = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        #if(len(files)>max):
        #    max = len(files)
        print(str(i) + " : " + str(len(files)))
        #sum = sum + len(files)
        filenum = 0
        for filename in files:
            print('files[0]' + str(files[0]))
            print('files[1]' + str(files[1]))
            print('files[2]' + str(files[2]))
            print(str(label_idx) + "class ," +str(filenum)+"filenum,"+ filename+"filename")
            img_path = os.path.join(root, filename)
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
            except:
                continue
            label_list.append(label_idx)
            img_list.append(img)
            filenum +=1 #
        label_idx += 1

    #print("max : " + str(max))
    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_list, img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_list, label_f)


def train_val_data_loader(data_path, img_size, output_path):
    label_list_train = []
    img_list_train = []
    label_list_val = []
    img_list_val = []
    label_idx = 0  # val이든 train이든 상관없이 매번 증가되어야만 함
    #i = 0
    val_send_check = False # 처음에 val이 들어갔는지
    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        print(str(label_idx) + " : " + str(len(files)))
        if (len(files) < 4):
            # 카테고리의 자료수가 4보다 적으면 val로 들어가지 않도록
            for filename in files:
                img_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(img_path, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                except:
                    continue
                label_list_train.append(label_idx)
                img_list_train.append(img)
                label_idx += 1

        else:
            # 카테고리의 자료수가 4이상이면 가장 먼저 것을 val_list에 옮기고,(continue) , 그다음 것들을 train에 들어가도록
            for filename in files:
                img_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(img_path, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                except:
                    continue
                if(val_send_check ==  False):
                    label_list_val.append(label_idx)
                    img_list_val.append(img)
                    val_send_check = True
                else:
                    label_list_train.append(label_idx)
                    img_list_train.append(img)
                label_idx += 1
            val_send_check = False




    # write output file for caching
    with open(output_path[0], 'wb') as train_f:
        pickle.dump(img_list_train, train_f)
    with open(output_path[1], 'wb') as val_f:
        pickle.dump(img_list_val, val_f)
    with open(output_path[2], 'wb') as t_lable_f:
        pickle.dump(label_list_train, t_lable_f)
    with open(output_path[3], 'wb') as v_label_f:
        pickle.dump(label_list_val,  v_label_f)


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
