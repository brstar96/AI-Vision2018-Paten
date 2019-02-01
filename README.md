#balacing 1  
    데이터를 balancing 하게 맞춘다.  
    iteration이 main.py에서 돌아간다.  
    output : img_list,label_list,class_list  

: main 코드 commit f0dd347, data_loader commit 49078ca 

#balacing 2  
    데이터를 balancing 하게 맞춘다.  
    iteration이 data_loader.py로 돌아간다. 
    output : img_list,label_list

: main 코드 commit 66656d6, data_loader commit 5d8d996

#balancing 3  
    2에서 iteration이 돌 때, 맨 마지막 iteration에서만 imread 한다.

    : data_loader commit 0463f5c

#balancning 4  
    디버깅 코드 삭제

    : data_loader commit 187ca7a
