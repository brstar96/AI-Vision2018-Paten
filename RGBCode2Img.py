import cv2
import numpy as np
import glob
import os
import itertools

txtPath = "./TextMining"

def txt2list(ImgCount, txtPath, output_Path):
    with open(txtPath) as f:
        txtList = f.readlines()
        txtList = str([x.strip() for x in txtList]) # 공백 제거
    for char in "_():.[]''": #특수문자 제거
        txtList = txtList.replace(char, "")
        txtList = txtList.replace(",", "")
        txtList = txtList.replace("  ", " ")
    newtxtList_int = [int(n) for n in txtList.split()] #int로 전부 형변환
    EmptyNpArr = np.asarray(newtxtList_int) #List를 numpy arr로 변환
    ImgNpArr = EmptyNpArr.flatten('F')
    TempArr = ImgNpArr.reshape(224, 224, 3)
    b,g,r = cv2.split(TempArr)
    rgb_img = cv2.merge([r,g,b])
    print(rgb_img)

    #이미지 출력
    cv2.imwrite(output_Path + "/" + "QueryImg" + str(ImgCount) +".jpg", rgb_img)

    # 테스트용 txt파일 작성
    # file = open("testfile.txt", "w")
    # for i in range(len(txtList)):
    #     file.write(txtList[i])
    # file.close()

if __name__ == '__main__':
    TextMining_referenceImgs = "./TextMining_referenceImgs" #레퍼런스 이미지의 row txt파일 경로
    TextMining_queryImgs = "./TextMining_queryImgs" #쿼리 이미지의 row txt파일 경로
    referenceImgs_output_Path = "./TextMiningResImgs_reference" #디코드된 레퍼런스 이미지가 저장되는 경로
    queryImgs_output_Path = "./TextMiningResImgs_query" #디코드된 쿼리 이미지가 저장되는 경로

    txtfilePathList = os.listdir(TextMining_queryImgs) #경로 내 모든 파일들의 디렉토리를 받아옴.
    for ImgCount in range(len(txtfilePathList)):
        txt2list(ImgCount, TextMining_queryImgs + "/" + txtfilePathList[ImgCount], queryImgs_output_Path)
