import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.
import seaborn as sns


# Seaborn : matplotlib의 기능 보강용 모듈
titanic = sns.load_dataset("titanic")
print(titanic.info())

sns.boxplot(y = "age", data = titanic, palette='Paired')  #타이타닉에서 age 데이터만 뽑아서 박스플롯을 만듬.
plt.show()

sns.displot(titanic['age']) #밀도
plt.show()
sns.kdeplot(titanic['age']) #막대
plt.show()

sns.relplot(x='who', y='age', data=titanic) #범주형
plt.show()

sns.countplot(x='class', data=titanic)
plt.show()

t_pivot = titanic.pivot_table(index='class', columns='sex', aggfunc='size')
print(t_pivot)

sns.heatmap(t_pivot, 
            cmap=sns.light_palette('gray', as_cmap=True), annot=True, fmt='d')
plt.show()
