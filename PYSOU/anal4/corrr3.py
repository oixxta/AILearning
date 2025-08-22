"""
외국인의 국내 주요 관광지 방문관련 상관관계 분석
"""
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

# scatter 차트 작성
def setScatterGraph(tour_table, all_table, tourPoint):
    #계산할 관광지 이름에 해당하는 자료만 뽑아 별도 저장하고 외국인 관광자료와 머지하기
    tour = tour_table[tour_table['resNm'] == tourPoint]
    # print(tour)
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    #print(merge_table)

    fig = plt.figure()
    fig.suptitle(tourPoint + '상관관계분석')
    plt.subplot(1, 3, 1)
    plt.xlabel('중국인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb1 = lambda p : merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    print('r1 : ', r1)
    plt.title("r={:.5f}".format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], alpha=0.7, s=6, c='red')

    plt.subplot(1, 3, 2)
    plt.xlabel('일본인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb2 = lambda p : merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    print('r2 : ', r2)
    plt.title("r={:.5f}".format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], alpha=0.7, s=6, c='blue')

    plt.subplot(1, 3, 3)
    plt.xlabel('미국인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb3 = lambda p : merge_table['america'].corr(merge_table['ForNum'])
    r3 = lamb3(merge_table)
    print('r3 : ', r3)
    plt.title("r={:.5f}".format(r3))
    plt.scatter(merge_table['america'], merge_table['ForNum'], alpha=0.7, s=6, c='green')

    plt.tight_layout()
    plt.show()
    plt.close()
    
    return [tourPoint, r1, r2, r3]

def departure():
    #서울시 관광지 정보 읽어서 Dataframe으로 저장
    fname = "서울특별시_관광지입장정보_2011_2016.json"
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'resNm', 'ForNum'))
    tour_table = tour_table.set_index('yyyymm')  #날짜별을 앞으로 테이블 바꾸기
    #print(tour_table)

    resNm = tour_table.resNm.unique()   #관광지 이름들 확인
    #print(resNm[:5])    #5개만 확인, ['창덕궁' '운현궁' '경복궁' '창경궁' '종묘']

    #중국인 관광 정보를 읽어 Dataframe에 저장
    cDf = "중국인방문객.json"
    jData = json.loads(open(cDf, 'r', encoding='utf-8').read())
    china_Table = pd.DataFrame(jData, columns=('yyyymm', 'visit_cnt'))
    china_Table = china_Table.rename(columns={'visit_cnt':'china'})
    china_Table = china_Table.set_index('yyyymm')
    #print(china_Table[:2])

    #일본인 관광정보
    jDf = "일본인방문객.json"
    jData = json.loads(open(jDf, 'r', encoding='utf-8').read())
    japan_Table = pd.DataFrame(jData, columns=('yyyymm', 'visit_cnt'))
    japan_Table = japan_Table.rename(columns={'visit_cnt':'japan'})
    japan_Table = japan_Table.set_index('yyyymm')
    #print(japan_Table[:2])

    #미국인 관광정보
    aDf = "미국인방문객.json"
    jData = json.loads(open(aDf, 'r', encoding='utf-8').read())
    america_Table = pd.DataFrame(jData, columns=('yyyymm', 'visit_cnt'))
    america_Table = america_Table.rename(columns={'visit_cnt':'america'})
    america_Table = america_Table.set_index('yyyymm')
    #print(america_Table[:2])

    #테이블들 머지
    all_table = pd.merge(china_Table, japan_Table, left_index=True, right_index=True)
    all_table = pd.merge(all_table, america_Table, left_index=True, right_index=True)
    #print(all_table)

    r_list = [] # 각 관광지별 상관계수를 저장할 리스트

    for tourPoint in resNm[:5]:
        #print(tourPoint)
        #각 관광지별 상관계수와 그래프 그리기
        r_list.append(setScatterGraph(tour_table, all_table, tourPoint))
    
    #r_list로 데이터프레임 작성
    r_df = pd.DataFrame(data=r_list, columns=('관광지', '중국인', '일본인', '미국인'))
    r_df = r_df.set_index('관광지')
    print(r_df)

    r_df.plot(kind='bar', rot=50)
    plt.show()
    plt.close()

if __name__ == '__main__':
    departure()