import cv2
import numpy as np
import glob
import os
import itertools

txtPath = "./TextMining"

def txt2list(ImgCount, txtPath, output_Path):
    EmptyNpArr = np.arange(0, 244 * 244 * 3)  # 3채널 244*244이미지어레이 생성

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
    cv2.imwrite(output_Path + "/" + "ReferenceImg" + str(ImgCount) +".jpg", rgb_img)

    # 테스트용 txt파일 작성
    # file = open("testfile.txt", "w")
    # for i in range(len(txtList)):
    #     file.write(txtList[i])
    # file.close()

if __name__ == '__main__':
    output_Path = "./TextMiningResImgs"
    txtfilePath = "./TextMining"
    txtfilePathList = os.listdir(txtfilePath) #경로 내 모든 파일들의 디렉토리를 받아옴.
    for ImgCount in range(len(txtfilePathList)):
        txt2list(ImgCount, txtfilePath +"/" +txtfilePathList[ImgCount], output_Path)
