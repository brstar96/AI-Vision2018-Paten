import cv2
import numpy as np
import os

def txt2list(mode, NumImgs, ImgCount, txtPath, output_Path, ImageScale):
    print(mode)
    WholeOneImgPixels = ImageScale[0] * ImageScale[1] * ImageScale[2]
    EntireImgPixels = WholeOneImgPixels * NumImgs
    print(EntireImgPixels)

    with open(txtPath) as FileHandler:
        txtList = FileHandler.readlines() # txt파일을 row 단위로 끊어 읽음
        txtList = str([x.strip() for x in txtList]) # 공백 제거
    for char in "_().:[]''": # "" 내의 특수문자 제거
        txtList = txtList.replace(char, "")
        txtList = txtList.replace(",", "")
        txtList = txtList.replace("  ", " ") # space가 2개인 경우를 1개로 줄임
    for i in range(NumImgs):  # 뽑아낼 이미지 장수만큼 Iteration
        if mode == "Reference":
            txtList = txtList.replace("referenceimg " + str(i), "")  # referenceimg 0과 같은 불필요 텍스트 삭제
        elif mode == "Query":
            txtList = txtList.replace("queryimg " + str(i), "")  # query_img 0과 같은 불필요 텍스트 삭제
        else:
            break

    for CurrentImageNum in range(EntireImgPixels / WholeOneImgPixels): #EntireImgPixels / WholeOneImgPixels 장수만큼 Iteration 수행
        addNum = WholeOneImgPixels * CurrentImageNum
        for i in range(WholeOneImgPixels):
            newList = []
            newList.append(txtList[addNum*WholeOneImgPixels])

        temp = CurrentImageNum * WholeOneImgPixels

    newtxtList_int = [int(n) for n in txtList.split()] # int로 전부 형변환
    print(type(newtxtList_int[0]))
    EmptyNpArr = np.asarray(newtxtList_int) # List를 numpy arr로 변환
    ImgNpArr = EmptyNpArr.flatten('F') # 확인차 한번 더 flatten
    TempArr = ImgNpArr.reshape(ImageScale[0], ImageScale[1], ImageScale[2]) # flatten된 arr를 224*224*3 numpy array로 reshape
    b,g,r = cv2.split(TempArr) # b, g, r 채널로 split
    rgb_img = cv2.merge([r,g,b]) # split된 채널을 r, g, b 순서로 merge
    print(rgb_img) # 테스트용 print

    # 이미지 출력
    if mode == 'Reference':
        cv2.imwrite(output_Path + "/" + "ReferenceImg" + str(ImgCount) +".jpg", rgb_img)
    else:
        cv2.imwrite(output_Path + "/" + "QueryImg" + str(ImgCount) + ".jpg", rgb_img)

    # 테스트용 txt파일 작성
    # file = open("testfile.txt", "w")
    # for i in range(len(txtList)):
    #     file.write(txtList[i])
    # file.close()

if __name__ == '__main__':
    TextMiningPath = "./VeryLongtxts" # Mining할 텍스트 파일의 경로
    TextMiningResImgPath_Reference = "./VeryLongtxtsResImgs/ReferenceImgs" # 디코드된 레퍼런스 이미지들이 저장되는 경로
    TextMiningResImgPath_Query = "./VeryLongtxtsResImgs/QueryImgs"  # 디코드된 레퍼런스 이미지들이 저장되는 경로
    ImageScale = [244, 244, 3] #이미지의 Width, Height, Channel
    NumRefImgs = 10  # 뽑아낼 reference image 장수
    NumQureyImgs = 10  # 뽑아낼 query image 장수

    txtfilePathList = os.listdir(TextMiningPath) # 경로 내 모든 파일들의 디렉토리를 받아옴.
    print(txtfilePathList)
    for ImgCount in range(len(txtfilePathList)):
        # def txt2list(mode, NumImgs, ImgCount, txtPath, output_Path):
        if ImgCount == 0:
            # Reference Img들을 출력
            mode = 'Query'
            txt2list(mode, NumRefImgs, ImgCount, TextMiningPath + "/" + txtfilePathList[ImgCount], TextMiningResImgPath_Reference, ImageScale)
        elif ImgCount == 1:
            # Query Img들을 출
            mode = 'Reference'
            txt2list(mode, NumQureyImgs, ImgCount, TextMiningPath + "/" + txtfilePathList[ImgCount], TextMiningResImgPath_Query, ImageScale)
        else:
            break

