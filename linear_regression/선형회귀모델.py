import numpy as np
import matplotlib.pyplot as plt

# 공부 시간 x와 성적 y의 넘파이 배열
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

plt.scatter(x, y)
plt.show()

a = 0
b = 0
lr = 0.03
epochs = 2001

n = len(x)

# 경사하강법
for i in range(epochs):
  y_pred = a * x + b  # 예측값
  error = y - y_pred  # 실제 값과 비교한 오차

  a_diff = (2/n) * sum(-x * error)  # 오차 함수를 a로 편미분
  b_diff = (2/n) * sum(-error)      # 오차 함수를 b를 편미분

  a = a - lr * a_diff   # 학습률을 곱해 기존의 a 값을 업데이트
  b = b - lr * b_diff   # 학습률을 곱해 기존의 b 값을 업데이트

  if i % 100 == 0:     # 100번 반복될 때마다 현재의 a 값, b 값을 출력
    print("epochs=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))

# 최종 a 값을 기울기, b 값을 y 절편에 대입해 그래프 생성
y_pred = a * x + b
plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.show()
