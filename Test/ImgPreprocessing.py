import os
import cv2
import glob
import itertools

'''
*NSML 데이터셋 경로 예시 
    -ir_ph1_v2/train/train_data/(라벨넘버)/s0.jpg (학습 데이터, 총 1000클래스 7104장)
    -ir_ph1_v2/test/test_data/query/s0.jpg (질의 이미지 폴더, 총 195장)
    -ir_ph1_v2/test/test_data/reference/s0.jpg (검색 대상 이미지 폴더, 총 1127장)
'''

#이미지 전처리
def Preprocessing_data_loader(res, img_size):
    img_list=[]
    img_idx=[]

    for i in range(len(res)):
        FileName = str(res[i])
        print(FileName)
        img_list.append(cv2.imread(FileName))
        img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB) #change channel input
        img_list[i] = cv2.resize(img_list[i], img_size) #224*224 resize

        im_height = img_list[i].shape[0]
        im_width = img_list[i].shape[1]
        print(im_height)
        print(im_width)

#file names를 가져오기
def getFileNames(exts):
    fnames = [glob.glob(ext) for ext in exts]
    fnames = list(itertools.chain.from_iterable(fnames))
    return fnames

def main():
    # PreprocessTestImgs 폴더에서 .png, .jpg를 가져오기
    exts = ["PreprocessTestImgs\*.png", "PreprocessTestImgs\*.jpg"]
    input_shape = (224, 224, 3)
    res = getFileNames(exts)
    Preprocessing_data_loader(res, input_shape[:2])

main()