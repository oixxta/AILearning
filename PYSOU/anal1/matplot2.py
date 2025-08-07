import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.


x = np.arange(10)

"""
# figure 구성 방법
# 1. matplotlib 스타일의 인터페이스 (plt.figure)
plt.figure()
plt.subplot(2, 1, 1)    # row, column, panel number
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()

# 2. 객체지향 인터페이스
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(np.random.randn(10), bins=10, alpha=0.9)   #히스토그램 그리기
ax2.plot(np.random.randn(10))
plt.show()

# 막대그래프 (bar)
data = [50, 80, 100, 70, 90]
plt.bar(range(len(data)), data)     #세로그래프
plt.show()

loss = np.random.rand(len(data))
plt.barh(range(len(data)), data)    #가로그래프
plt.show()

loss = np.random.rand(len(data))
plt.barh(range(len(data)), data, xerr=loss)    #가로그래프
plt.show()

# pie그래프
plt.pie(data)   #파이그래프는 모양 특성상 데이터 양이 적어야만 표현 가능!
plt.show()
plt.pie(data, explode=(0, 0.1, 0, 0, 0), colors=['yellow', 'red', 'blue'])
plt.show()

# boxplot   : 사분위 등에 대한 데이터 분포 확인에 대단히 효과적임!
plt.boxplot(data)
plt.show()

"""

# 버블 차트 : 데이터의 크기에 따라 원의 크기가 커짐.
n = 30
np.random.seed(0)
x = np.random.rand(n)
y = np.random.rand(n)
color = np.random.rand(n)
scale = np.pi * (15 * np.random.rand(n)) ** 2
plt.scatter(x, y, c = color, s = scale)
plt.show()

#시리얼 데이터
import pandas as pd
fdata = pd.DataFrame(np.random.randn(1000, 4), 
                     index = pd.date_range('1/1/2000', periods=1000), columns=list('ABCD'))
fdata = fdata.cumsum()
print(fdata.head(3))
plt.plot(fdata)
plt.show()

#판다스 자체에서 지원하는 plot, matplotlib 없이도 시각화가 가능함.
fdata.plot()
fdata.plot(kind='bar')
fdata.plot(kind='box')
plt.xlabel("time")
plt.ylabel("data")
plt.show()