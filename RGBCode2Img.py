import cv2
import numpy as np

txtPath = "./TextMining"


def txt2list(txtPath):
    EmptyNpArr = np.arange(0, 244 * 244 * 3)  # 3채널 244*244이미지어레이 생성

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
    TempArr = ImgNpArr.reshape(224, 224, 3)
    b,g,r = cv2.split(TempArr)
    rgb_img = cv2.merge([r,g,b])
    print(rgb_img)

    #이미지 출력
    cv2.imwrite('./query1.jpg', rgb_img)

    # 테스트용 txt파일
    # file = open("testfile.txt", "w")
    # for i in range(len(txtList)):
    #     file.write(txtList[i])
    # file.close()

    #Numpy Array test용 txt파일
    # filw = open()
txt2list(txtPath)