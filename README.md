1. load시 최근에 진행했던 checkpoint와 session 이름을 nsml.load 메소드에서 수정한다.
2. 마지막 코드부분에 checkpoint의 test 이름을 아래 4번의 -m 명령어와 통일시킨다.
3. 학습시작 시간 기록해둘 것
4. 학습 돌리기    
    > nsml에 로그인된 상태에서 nsml run -d ir_ph1_v2 -g 1 -m 'default_densenet_model_1_' -e main.py 입력

        - 첨자 목록은 https://n-clair.github.io/vision-docs/_build/html/ko_KR/contents/session/run_a_session.html 참고 
5. Inference - 학습이 완료된 상태에서 수행 가능, 한번 submit하면 1시간 뒤에 submit 가능.  
    > 1) 1시간패널티 있음 : nsml submit team_33/ir_ph1_v2/25 default_densenet_model_1_990    
    > 2) 패널티 없음 : nsml submit -t team_33/ir_ph1_v2/25 default_densenet_model_1_990 
    
        - 첨자 목록은 https://n-clair.github.io/vision-docs/_build/html/ko_KR/contents/session/submit_a_session.html?highlight=submit 참고
        - Submit을 수행하면 nsml run시에 올라간 main.py 내의 infer() 함수가 실행됨. 

+Learning rate도 기록할것.(추후)

