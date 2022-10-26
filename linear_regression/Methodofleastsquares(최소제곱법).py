import numpy as np
from matplotlib import pyplot as plt

# 공부 시간 x와 성적 y의 넘파이 배열
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

# x, y 의 평균값
mx = np.mean(x)
my = np.mean(y)
print("mean x:", mx)
print("mean y:", my)

# 기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])

# 기울기 공식의 분자
def top(x, mx, y, my):
  d = 0
  for i in range(len(x)):
    d += (x[i] - mx) * (y[i] - my)
  return d

dividend = top(x, mx, y, my)

print("분모:", divisor)
print("분자:", dividend)

# 기울기와 y절편
a = dividend / divisor
b = my - (mx * a)

# 출력값 최종 확인
print("기울기 a =", a)
print("y 절편 b =", b)

plt.plot(x, y, marker='o', linestyle='dashed')
plt.show()
