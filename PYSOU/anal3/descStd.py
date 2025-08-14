# 표준편차, 분산을 중요.
# 두 반의 시험 성적이 "평균이 같다고 해서 성적분포가 동일한가?"를 증명
# 이를 확인하려면 표준편차와 분산값도 함계 확인해야 함.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.

np.random.seed(42)  #데이터로 쓸 랜덤값을 고정 : 42번 시드로
#목표 평균 ...
target_mean = 60    #목표 평균값
std_dev_small = 10  #제일 작은 표쥰편차
std_dev_big = 20    #제일 큰 표준편차

class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size=100)
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_big, size=100)

#평균값 보정
class1_adj = class1_raw - np.mean(class1_raw) + target_mean
class2_adj = class2_raw - np.mean(class2_raw) + target_mean

#정수화 및 범위 제한
class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)
print(class1)
print(class2)

#통계값 계산
mean1, mean2 = np.mean(class1), np.mean(class2)
std1, std2 = np.std(class1), np.std(class2)
var1, var2 = np.var(class1), np.var(class2)

print("1반 성적")
#print(class1)
print(f"평균 : {mean1:.2f}, 표준편차 : {std1:.2f}, 분산 : {var1:.2f},")
print("2반 성적")
#print(class2)
print(f"평균 : {mean2:.2f}, 표준편차 : {std2:.2f}, 분산 : {var2:.2f},")

"""
표준편차와 분산값을 비교해본 결과, 평균이 비슷한 반이라도 학생들의 학업 수준이 비슷하다고 단정할 수 없음.
"""


df = pd.DataFrame({
    'class' : ['1반'] * 100 + ['2반'] * 100,
    'score' : np.concatenate([class1, class2])
})
print(df.head(3))
print(df.tail(3))
df.to_csv('desc_std1_1.csv', index=False, encoding='utf-8')    #외부파일로 저장


# 시각화 : 데이터분석에 시각화는 필수임. (산포도 혹은 박스플롯이 일반적임)
#산포도
x1 = np.random.normal(1, 0.05, size=100)
x2 = np.random.normal(1, 0.05, size=100)

plt.figure(figsize=(10, 6))
plt.scatter(x1, class1, label=f'1반 (평균 = {mean1:.2f}, σ = {std1:.2f})')
plt.scatter(x2, class2, label=f'2반 (평균 = {mean2:.2f}, σ = {std2:.2f})')
plt.hlines(target_mean, 0.5, 2.5, colors='red', linestyles='dashed', label=f'공통평균={target_mean}')
plt.title('동일 평균, 상이 성적분포를 가진 두 반 비교')
plt.xticks([1, 2], ['1반', '2반'])
plt.ylabel('시험 점수')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

#박스플롯
plt.figure(figsize=(8, 5))
plt.boxplot([class1, class2], labels=['1반', '2반'])
plt.title('성적 분포를 가진 두 반 비교')
plt.ylabel('시험 점수')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close() #메모리 관리를 위한 plt 종료