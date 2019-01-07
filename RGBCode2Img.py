import cv2
import numpy as np

txtPath = "./TextMining"


def txt2list(txtPath):
    EmptyNpArr = np.arange(0, 244 * 244 * 3)  # 3채널 244*244이미지어레이 생성
    # print(len(EmptyNpArr))
    imgWidth = 224
    imgHeight = 224
    imgChannel = 3

    with open(txtPath + '/reference_imgs_row.txt') as f:
        txtList = f.readlines()
        txtList = str([x.strip() for x in txtList]) # 공백 제거
    for char in "_():.[]''": #특수문자 제거
        txtList = txtList.replace(char, "")
        txtList = txtList.replace(",", "")
        txtList = txtList.replace("  ", " ")
    newtxtList_int = [int(n) for n in txtList.split()] #int로 전부 형변환
    EmptyNpArr = np.asarray(newtxtList_int) #List를 numpy arr로 변환
    ImgNpArr = EmptyNpArr.flatten('F')
    # print(ImgNpArr.reshape(224, 224, 3))

    #이미지 출력
    # cv2.imwrite('query1.jpg', ImgNpArr.reshape(224, 224, 3))
    cv2.imshow('query1.jpg', ImgNpArr.reshape(224, 224, 3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 테스트용 txt파일
    # file = open("testfile.txt", "w")
    # for i in range(len(txtList)):
    #     file.write(txtList[i])
    # file.close()

txt2list(txtPath)