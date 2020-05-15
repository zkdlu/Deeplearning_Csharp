# CS_Deeplearning_without_library


입력 - 은닉 - 은닉 - 출력 계층으로 이루어진 인공 신경망을 C#으로 만들어보았다.
- 출력층에서는 활성화 함수 Sigmoid를 사용하고 은닉층에서는 ReLU를 사용
- 이전층에서 입력되어진 값과 weight들의 합산에 대한 미분 방정식을 세워 경사하강법으로 역전파를 수행
