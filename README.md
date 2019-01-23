사용방법 : 업로드한 data_loader.py 변경하고 main.py 아래 사용법에 맞게 수정해서 사용하면 됩니다~ 

1. data_loader.py 수정
2. train_val_data_loader 함수 추가 
역할 : train data와 validation data를 분리해서 불러옴
현재 한 클래스의 4개이상의 이미지가 존재하면 validation을 뽑아내도록 되어있음 
4개가 아닌 다른 숫자로 설정하려면 
함수 내의 
 if (len(files) < 4):
            # 카테고리의 자료수가 4보다 적으면 val로 들어가지 않도록
            for filename in files:
 4 숫자를 변경
 
 3. pickle 자료형으로 나오는 것 총 4개, 
     1) train 이미지들이 파이썬 리스트 객체로 저장되어있는 train_f 
     2) validation으로 뽑아낸 이미지들이 파이썬 리스트 객체로 저장되어있는 val_f 
     3) train 라벨 데이터들이 파이썬 리스트 객체로 저장되어있는 t_lable_f
     4) validation 라벨 데이터들이 파이썬 리스트 객체로 저장되어있는 v_label_f

4. main.py에서 다음을 수정 
(1) 만들어놓은 train_val_data_loader 모듈 가져오기
    from data_loader import train_data_loader, train_val_data_loader

(2) data_loader 바꾸기 
<기존>
 if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
                       output_path=output_path)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
            train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)
            
 
 nsml.cache 부분과 , else에 있는 train_data_loader를 train_val_data_loader로 수정
 
 (3) 읽어오는 pickle 부분 수정
 
       #아래 과정에서 읽기 시작함
        #pickle (현재 메모리에 살아있는 ?) 파이썬 객체 자체를 읽고 저장하기 위해
        #type 그대로 python에서 만들어지는 모든 것들예) array([ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318,  0.4236548 ,]
        #with 문 자동으로 파일 close 해줌
        with open(output_path[0], 'rb') as train_f:
            img_list_train = pickle.load(train_f)
        with open(output_path[1], 'rb') as val_f:
            img_list_val = pickle.load(val_f)
        with open(output_path[2], 'rb') as t_label_f:
            label_list_train = pickle.load(t_label_f)
        with open(output_path[3], 'rb') as v_label_f:
            label_list_val = pickle.load(v_label_f)

  
        x_train = np.asarray(img_list_train )
        t_labels = np.asarray(label_list_train )
        x_val = np.asarray(img_list_val)
        v_labels = np.asarray(label_list_val)
     


        print(x_train.shape, ' x_train shape') # (전체 train 개수 - val로 빠진 개수, 224, 224,3) 
        print(t_labels.shape,' t_labels shape') #(전체 train 개수 - val로 빠진 개수,)
        print(x_val.shape, ' x_val shape') #(val로 빠진 개수, 224,224,3)
        print(v_labels.shape, ' v_labelsshape') #(val로 빠진 개수,)
        
        이후는  cross validation 하는 부분 삭제하고 원래 train 방식에 맞게 진행 
 



