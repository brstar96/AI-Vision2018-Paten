import os
import numpy as np
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
def Preprocessing_data_loader(ImgNameList, img_size, output_path):
    for ImgNo in range(0, len(ImgNameList)):
        FileName = str(ImgNameList[ImgNo])
        Img = cv2.imread(FileName)
        Img = cv2.resize(Img, img_size)  # 224*224 resize
        ImgHeight, ImgWidth, _ = Img.shape
        templist = [[[0 for col in range(3)] for row in range(img_size[0])] for rgb in range(img_size[0])]

        newImage = np.array(templist) #224 * 224 *3 새 이미지 생성
        NewImgHeight, NewImgWidth,_ = newImage.shape
        # print("New Image Shape :", newImage.shape)

        for height in range(0,NewImgHeight):
            for width in range(0,NewImgWidth):
                newImage[height][width] = Img[height][width]

        cv2.imwrite(output_path+ "/" + str(ImgNo) + ".jpg", newImage)
        # print("file has been created : ", str(ImgNo) + ".jpg")

#file names를 가져오기
def getFileNames(exts):
    fnames = [glob.glob(ext) for ext in exts]
    fnames = list(itertools.chain.from_iterable(fnames))
    # print(fnames)
    return fnames

if __name__ == '__main__':
    # PreprocessTestImgs 폴더에서 .png, .jpg를 가져오기
    output_path = "./Changed_Data_example_ph1"
    DatasetPath = "Data_example_ph1"
    print(os.listdir(DatasetPath))
    exts = [DatasetPath + "/" + os.listdir(DatasetPath)[0] + "/" + "*.jpg"]
    print(exts)


    input_shape = (224, 224, 3)
    ImgNameList = getFileNames(exts)
    print(ImgNameList)

    Preprocessing_data_loader(ImgNameList, input_shape[:2], output_path)


