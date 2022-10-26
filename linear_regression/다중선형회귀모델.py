import numpy as np
import matplotlib.pyplot as plt

# 공부 시간 x1과 과외 시간 x2, 성적 y의 넘파이 배열
x1 = np.array([2, 4, 6, 8])
x2 = np.array([0, 4, 2, 3])
y = np.array([81, 93, 91, 97])

# 데이터의 분포 그래프
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1, x2, y)
plt.show()

a1 = 0
a2 = 0
b = 0
lr = 0.01
epochs = 2001

n = len(x1)

# 경사 하강법
for i in range(epochs):
  y_pred = a1 * x1 + a2 * x2 + b    # 예측값을 구하는 식
  error = y - y_pred                # 실제 값과 비교한 오차

  a1_diff = (2/n) * sum(-x1 * error)  # 오차 함수를 a1로 편미분
  a2_diff = (2/n) * sum(-x2 * error)  # 오차 함수를 a2로 편미분
  b_diff = (2/n) * sum(-error)        # 오차 함수를 b로 편미분

  a1 = a1 - lr * a1_diff      # 학습률을 곱해 기존의 a1 값을 업데이트
  a2 = a2 - lr * a2_diff      # 학습률을 곱해 기존의 a2 값을 업데이트
  b = b - lr * b_diff         # 학습률을 곱해 기존의 b 값을 업데이트

  if i % 100 == 0:
    print("epochs=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" % (i, a1, a2, b))

# 실제 점수와 예측된 점수를 출력
print("실제 점수:", y)
print("예측 점수:", y_pred)
