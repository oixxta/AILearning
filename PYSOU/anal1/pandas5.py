import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#파일 입출력
#외부 CSV파일 읽어오기
df = pd.read_csv('./ex1.csv')
print(df, type(df))                 #CSV를 바로 판다스의 데이터프레임으로 바꿔줌
print(df.info())                    #만들어진 데이터프레임 자료형 정보들
df = pd.read_csv('./ex1.csv', sep=',')  #구분자가 공백이나 다른 것일경우, sep의 값을 다른것으로 하면 문제없음!

df = pd.read_table('./ex1.csv') #CSV 파일을 read_table로 읽을 순 있지만 문자열 그대로 읽힘!
print(df)
df = pd.read_table('./ex1.csv', sep=',') #따라서 이 경우에는 반드시 sep로 구분자를 지정해 줘야함!
print(df)

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
print(df) #URL도 읽을 수 있음.
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None)
print(df)   #Header를 none으로 지정하면 열 이름을 사용하지 않음.
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                 header=None, names=['a', 'b', 'c', 'd', 'msg'])
print(df)   #열 이름을 오른쪽부터 커스텀 네임으로 지정 가능!
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                 header=None, names=['a', 'b', 'c', 'd', 'msg'], skiprows=1)
print(df)   #skiprows로 특정 줄 스킵 가능

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt')
print(df)           #txt 파일도 csv로 읽을수는 있지만,
print(df.info())    #모든 내용을 그냥 한줄로 읽어버리기 때문에 사용 불가능함.
df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt',
                   sep='\s+')
print(df)           
print(df.info())    #따라서 정규표현식을 활용해서 가공해야 함.

#자리수는 일치하지만, 콤마도, 공백도 없는 데이터를 읽어와 데이터프레임으로 만들기
df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt', 
                 widths=(10,3,5), names=['date', 'name', 'price'], encoding='utf')
print(df)   #widths로 몇 자리마다 끊어 읽을지 지정 가능!

"""
print()
url = "https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4"
df = pd.read_html(url)
print(df)
print(f"총 {len(df)}개의 자료")     #웹에 있는 자료 긁어오는 법
"""

#엄청난 량의 자료를 읽어야 하지만, RAM이 부족할 경우 청크(Chunk)로 끊어서 읽어야 함.
#대량의 데이터 파일을 읽는 경우, chunk 단위로 분리해 읽기 가능.
#효과1. 메모리 절약,
#효과2. 스트리밍 방식으로 순차 처리가 가능함.(+ 로그 분석, 실시간 데이터 처리, 머신러닝의 데이터 처리...)
#효과3. 분산 처리(batch)
#단점 : 한번에 다 읽는 것 보다는 처리 속도가 떨어짐.
import time
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic')
n_rows = 10000
"""
data = {
    'id': range(1, n_rows + 1),
    'name':[f'Stuednt_{i}' for i in range(1, n_rows + 1)],
    'score1': np.random.randint(50, 101, size=n_rows),
    'score2': np.random.randint(50, 101, size=n_rows)
}
df = DataFrame(data)
print(df.head(3))
print(df.tail(3))
csv_path = 'students.csv'
df.to_csv(csv_path, index = False)
"""
#작성된 csv 파일 사용 : 전체 모두를 읽기
start_all = time.time()   #현재시간 저장
df_all = pd.read_csv('./students.csv')
#print(df_all)
average_all_1 = df_all['score1'].mean()
average_all_2 = df_all['score2'].mean()
time_all = time.time() - start_all
print(time_all)

#Chunk로 처리
chunk_size = 1000   #1000개만 처리
total_score1 = 0 
total_score2 = 0
total_count = 0
start_chunk_total = time.time()
for i, chunk in enumerate(pd.read_csv('./students.csv', chunksize = chunk_size)):
    start_chunk = time.time()
    # 청크 처리할 때 마다 첫 번째 학생 정보만 출력
    first_student = chunk.iloc[0]
    print(first_student['id'], first_student['name'])
    total_score1 += chunk['score1'].sum()
    total_score2 += chunk['score2'].sum()
    total_count += len(chunk)
    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f"    처리 시간 : {elapsed}초")
time_chunk_total = time.time() - start_chunk_total
average_chunk_1 = total_score1 / total_count
average_chunk_2 = total_score2 / total_count

print('\n처리 결과 요약')
print(f'전체학생수 : {total_count}')
print(f'score1 총합 : {total_score1}, 평균 : {average_chunk_1:.4f}')
print(f'score2 총합 : {total_score2}, 평균 : {average_chunk_2:.4f}')

print(f'전체 한번에 처리한 경우 소요시간 : {time_all:.4f}')
print(f'청크로 처리한 경우 소요시간 : {time_chunk_total:.4f}')

#시각화
labels = ['전체 한번에 처리', '청크로 처리']
times = [time_all, time_chunk_total]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=['skyblue', 'yellow'])

for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{time_val:.4f}초', ha='center', va='bottom', fontsize=10)

plt.ylabel('처리시간(초)')
plt.title('전체 한번에 처리 vs 청크로 처리')
plt.grid(alpha = 0.5)
plt.tight_layout()
plt.show()