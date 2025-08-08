#워싱턴 D.C. 자전거 공유 시스템 파일로 시각화
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.


plt.style.use('ggplot')
#데이터 수집 후 가공(EDA) - train.csv를 바로 쓸거기 때문에
train = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv', parse_dates=['datetime'])
print(train.shape)
print(train.columns)
print(train.info()) #datetime만 오브젝트(문자열), 나머지들은 숫자(int64, float64). 따라서 datetime은 가공이 필요.(parse_dates=['datetime'])
print(train.head(3))
pd.set_option('display.max_columns', 500)   #터미널 창 안짤리고 다 보게 하는 법
print(train.describe())   #요약통계
print(train.temp.describe())    #temp 칼럼에 대한 요약통계만 보기
print(train.isnull().sum())     #null값 존재 여부 확인
"""
#null이 포함된 열 확인용 시각화 모듈(missingno)
import missingno as msno
msno.matrix(train, figsize=(12, 5))
plt.show()
msno.bar(train)
plt.show()
"""

#연월일시 데이터로 자전거 대여량을 시각화
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second       #데이트타임 칼럼을 이용해 연월일시분초 칼럼을 추가시킴.
print(train.columns)
print(train.head(1))

figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)   #시각화
figure.set_size_inches(15, 5)
sns.barplot(data=train, x='year', y='count', ax=ax1)
sns.barplot(data=train, x='month', y='count', ax=ax2)
sns.barplot(data=train, x='day', y='count', ax=ax3)
sns.barplot(data=train, x='hour', y='count', ax=ax4)
ax1.set(ylabel='건수', title='년도별 대여량')
ax2.set(ylabel='월', title='월 대여량')
ax3.set(ylabel='일', title='일 대여량')
ax4.set(ylabel='시', title='시 대여량')
plt.show()


#Bioxplot으로 시각화 - 대여량 - 계절별, 시간별 근무일 여부에 따른 대여량
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(x = 'season', y = 'count', data = train, ax=axes[0][0])
sns.boxplot(x = 'hour', y = 'count', data = train, ax=axes[0][1])
sns.boxplot(x = 'workingday', y = 'count', data = train, ax=axes[1][1])
plt.show()
