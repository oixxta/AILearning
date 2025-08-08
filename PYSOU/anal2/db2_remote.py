#원격데이터 연동 후 자료를 읽어 데이터프레임에 저장

#MySQLClient 설치 필요 : pip install MySQLClient
import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.
import sys
import pickle
import csv

"""
conn = MySQLdb.connect(
    host='127.0.0.1',
    user='root',
    password = '1234',
    database = 'mydb',
    port = 3306,
    charset = 'utf8'
)
"""
try:
    with open('myMaria.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('읽기 오류 : myMaria', e)
    sys.exit()


try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    sql = """
        select jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
    """
    cursor.execute(sql)
    #출력1 : console로 출력
    for(jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay) in cursor:
        print(jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay)
    print()
    #출력2 : 데이터프레임으로 출력
    df1 = pd.DataFrame(cursor.fetchall(), 
                       columns=['jikwonno', 'jikwonname', 'busername', 'jikwonjik', 'jikwongen', 'jikwonpay'])
    print(df1.head(3))
    #출력3 : CSV 파일로 출력
    with open('jik_data.csv', mode = 'w', encoding='utf-8') as obj:
        writer = csv.writer(obj)
        for r in cursor:
            writer.writerow(r)
    #3에서 만든 CSV 파일을 다시 데이터프레임에 저장
    df2 = pd.read_csv('jik_data.csv', header=None, names=['번호', '이름', '부서', '직급', '성별', '연봉'])
    print(df2.head(3))
    """
    #DB의 자료를 pandas의 SQL 처리 기능으로 읽기
    df = pd.read_csv(sql, conn)
    df.columns = ['번호', '이름', '부서', '직급', '성별', '연봉']
    print(df.head(3))
    """
    #DB의 자료를 데이터프레임으로 읽었음으로, pandas의 기능을 적용할 수 있음.
    print('건수 : ', len(df2))
    print('건수 : ', df2['이름'].count())
    print('직급별 인원수 : ', df2['직급'].value_counts())
    print('연봉 평균 : ', df2.loc[:, '연봉'].mean())
    print()
    ctab = pd.crosstab(df2['성별'], df2['직급'], margins=True)    #성별 & 직급별 건수, 크로스테이블
    print(ctab.to_html) #해당 테이블을 HTML로 저장

    #시각화
    #직급별 연봉 평균에 대한 시각화 - pie 그래프
    jik_ypay = df2.groupby(['직급'])['연봉'].mean()
    print('직급별 연봉 평균 : ', jik_ypay)
    print(jik_ypay.index)
    print(jik_ypay.values)

    plt.pie(jik_ypay, explode=(0.2, 0, 0, 0.3, 0), labels=jik_ypay.index, shadow=True, labeldistance=0.7,
            counterclock=False)
    plt.show()


except Exception as e:
    print('fail to connect!', e)
    
finally:
    conn.close()