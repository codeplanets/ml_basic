import numpy as np
import matplotlib.pyplot as plt

# Tensorflow-Keras API에서 필요한 함수 import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

model = Sequential()

# 출력 값, 입력 변수, 분석 방법에 맞게끔 모델을 설정
model.add(Dense(1, input_dim=1, activation='linear'))

# 오차 수정을 위해 경사 하강법(SGD)을, 오차의 정도를 판단하기 위해 평균 제곱 오차(MSE)를 사용
model.compile(optimizer='sgd', loss='mse')

# 오차를 최소화하는 과정을 2000번 반복
model.fit(x, y, epochs=2000)

plt.scatter(x, y)
plt.plot(x, model.predict(x), 'r')    # 예측 결과 그래프
plt.show()

# 임의의 시간을 집어넣어 점수를 예측하는 모델 테스트
hour = 7
prediction = model.predict([hour])
print("%.f시간을 공부할 경우의 예상 점수는 %.02f점입니다." % (hour, prediction))
