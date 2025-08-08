#로컬 데이터베이스 연동 후 자료를 읽어 데이터프레임에 저장하기

import sqlite3

sql = "create table if not exists test(procuct varchar(10), maker varchar(10), weight real, price integer)"
conn = sqlite3.connect(':memory:')  #테스트데이터, 램에만 작동하고 저장 안함!
conn.execute(sql)
conn.commit()       #커밋까지 해야 생성이 끝남.

#한개씩 추가
stmt = "insert into test values(?, ?, ?, ?)"    #다음과 같이 미리 매핑을 시켜야 시큐어코딩이 가능함.
data1 = ('mouse1', 'samsung', 12.5, 5000)    #튜플과 리스트 둘 다 가능함.
conn.execute(stmt, data1)
data2 = ('mouse2', 'samsung', 12.5, 8000)
conn.execute(stmt, data2)

#복수의 데이터 추가
datas = [('mouse3', 'LG', 22.5, 15000), ('mouse4', 'LG', 25.5, 15500)]
conn.executemany(stmt, datas)

cursor = conn.execute("select * from test")
rows = cursor.fetchall()
print(rows[0], '', rows[1], rows[0][0])
for a in rows:
    print(a)

#DB를 데이터프레임에 넣기
import pandas as pd
#방법1
df = pd.DataFrame(rows, columns=['product', 'maker', 'weight', 'price'])
print(df)
#print(df.to_html())

#방법2
df2 = pd.read_sql("select * from test", conn)   #이렇게 하면 fetchall을 할 필요 없음!
print(df2)

#데이터프레임의 내용을 DB에 넣기
pdata = {
    'product' : ['연필', '볼펜', '지우개'],
    'maker' : ['동아', '모나미', '모나미'],
    'weight' : [1.5, 5.5, 10.0],
    'price' : [500, 1000, 1500]
}
frame = pd.DataFrame(pdata) #데이터프레임 제작
#print(frame)
frame.to_sql("test", conn, if_exists='append', index=False)

df3 = pd.read_sql("select product, maker, price, weight as 무게 from test", conn)
print(df3)



#데이터베이스 작업이 끝나면 다음 두 가지는 반드시 실행해야함.
cursor.close()
conn.close()      


