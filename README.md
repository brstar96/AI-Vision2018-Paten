1. load시 최근에 진행했던 checkpoint와 session 이름을 nsml.load 메소드에서 수정한다.
2. 마지막 코드부분에 checkpoint의 test 이름을 아래 4번의 -m 명령어와 통일시킨다.
3. 학습시작 시간 기록해둘 것
4. 학습 돌리기

    1)nsml run -d ir_ph1_v2 -g 1 -m 'default_densenet_model_1_' -e main.py
5. Inference

    1)1시간패널티 있음 : nsml submit team_33/ir_ph1_v2/25 default_densenet_model_1_990
    
    2)패널티 없음 : nsml submit -t team_33/ir_ph1_v2/25 default_densenet_model_1_990

+Learning rate도 기록할것.(추후)