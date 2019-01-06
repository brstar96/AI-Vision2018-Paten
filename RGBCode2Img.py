import cv2
import numpy as np

txtPath = "./TextMining"
imgWidth, imgHeight = 244
imgChannel = 3
NewImg = np.zeros(imgWidth,imgHeight,imgChannel) #빈 3채널 244*244이미지어레이 생성

def txt2list(txtPath):
    with open(txtPath + '/reference_imgs_row.txt') as f:
        txtList = f.readlines()
        txtList = str([x.strip() for x in txtList]) # 공백 제거
    for char in "_():.[]''":
        txtList = txtList.replace(char, "")
        newtxtList = txtList.split(", ")
        # print(newtxtList)
    for i, val in enumerate(newtxtList):
        newtxtList[i] = newtxtList[i].split()
        print(newtxtList[i])

    for i in range(imgChannel):
        for ImgWidth in range(imgWidth):
            for imgHeight in range(imgHeight):
                # NewImg np array에 newtxtList[2][1]처럼 픽셀값 대입하는부분 작성할것.

    #테스트용 txt파일
    file = open("testfile.txt", "w")
    for i in range(len(txtList)):
        file.write(txtList[i])
    file.close()

txt2list(txtPath)