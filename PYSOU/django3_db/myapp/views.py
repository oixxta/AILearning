from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd
import datetime
from pathlib import Path
from django.conf import settings
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.

# Create your views here.

def indexFunc(request):
    return render(request, 'index.html')

def dbshowFunc(request):
    dept = request.GET.get('dept', "").strip()
    # inner join
    sql = """
        SELECT j.jikwonno as 직원번호, j.jikwonname as 직원명,
               b.busername as 부서명, b.busertel as 부서전화번호,
               j.jikwonpay as 연봉, j.jikwonjik as 직급
        FROM jikwon j INNER JOIN buser b
        ON j.busernum = b.buserno
    """
    params = []
    if dept:
        sql += " WHERE b.busername LIKE %s"
        params.append(f"%{dept}%")  # SQL 인젝션 해킹 방지(시큐어 코딩)
    sql += " ORDER BY j.jikwonno"

    with connection.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        #print('cur.description : ', cur.description)

        cols = [c[0] for c in cur.description]
        #print('cols : ', cols)

        df = pd.DataFrame(data=rows, columns=cols)
        #print(df.head(3))

        #Join 결과로 HTML 생성
        if not df.empty:
            join_html = df[['직원번호', '직원명', '부서명', '부서전화번호', '연봉', '직급']].to_html(index=False)
        else:
            join_html = '조회된 자료가 없습니다!'
        
        #직급별 연봉 통계표 (NaN은 0으로 처리)
        if not df.empty:
            stats_df = (
                df.groupby("직급")["연봉"]
                    .agg(평균="mean", 표준편차=lambda x:x.std(ddof=0), 인원수="count")
                    .round(2)
                    .reset_index()
                    .sort_values(by="평균", ascending = False)
            )
            stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
            stats_html = stats_df.to_html(index = False)
        else:
            stats_html = "통계 대상 자료가 없음!"
        
        ctx_dict = {
            'dept' : escape(dept),          # escape는 문자열에 특수문자가 있는 경우 HTML 엔티티로 치환하는 역할(단순문자 취급, 해킹방지)
                                            # 예) escape('<script>alert(1)</script>' -> '&lt;script&gt;...')
            'join_html' : join_html,
            'stats_html' : stats_html,
        }
    
    return render(request, 'dbshow.html', ctx_dict)

"""
MariaDB에 저장된 jikwon, buser 테이블을 이용하여 아래의 문제에 답하시오.
Django 모듈을 사용하여 결과를 클라이언트 브라우저로 출력하시오.

1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
       : 부서번호, 직원명 순으로 오름 차순 정렬 
2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.

3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.

4) 성별, 직급별 빈도표를 출력하시오.
"""

def homeworkFunc(request):
    sql = """
        SELECT j.jikwonno as 직원번호, j.jikwonname as 직원명,
               b.busername as 부서명, j.jikwonjik as 직급,
               j.jikwonpay as 연봉, j.jikwonibsail as 근속년수,
               j.jikwongen as 성별
        FROM jikwon j 
        INNER JOIN buser b ON j.busernum = b.buserno
    """
    with connection.cursor() as cur:
        #1. 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. : 부서번호, 직원명 순으로 오름 차순 정렬 
        cur.execute(sql)
        rows = cur.fetchall()
        #print(cur.description)
        cols = [c[0] for c in cur.description]
        #print(cols)
        mydf = pd.DataFrame(data=rows, columns=cols)
        #print(mydf.head(3))
        mydf['근속년수'] = pd.to_datetime(mydf['근속년수'])
        mydf['근속년수'] = datetime.datetime.now().year - mydf['근속년수'].dt.year
        mydf_sorted1 = mydf.sort_values(by = ['부서명'])
        mydf_sorted2 = mydf.sort_values(by = ['직원명'])

        #2. 부서명, 직급 자료를 이용하여 각각 연봉합, 연봉평균을 구하시오.
        mydf_fillterd1 = mydf[mydf['부서명'] == '총무부']
        gadPaySum = mydf_fillterd1['연봉'].sum()
        gadPayMean = mydf_fillterd1['연봉'].mean()
        mydf_fillterd2 = mydf[mydf['부서명'] == '영업부']
        sdPaySum = mydf_fillterd2['연봉'].sum()
        sdPayMean = mydf_fillterd2['연봉'].mean()
        mydf_fillterd3 = mydf[mydf['부서명'] == '전산부']
        cdPaySum = mydf_fillterd3['연봉'].sum()
        cdPayMean = mydf_fillterd3['연봉'].mean()
        mydf_fillterd4 = mydf[mydf['부서명'] == '관리부']
        adPaySum = mydf_fillterd4['연봉'].sum()
        adPayMean = mydf_fillterd4['연봉'].mean()

        #3. 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
        static_app_dir = Path(settings.BASE_DIR) / 'static'
        static_app_dir.mkdir(parents=True, exist_ok=True)
        img_path = static_app_dir / 'myGraph.png'
        plt.figure()
        data = mydf.groupby('부서명')['연봉'].sum()
        plt.bar(data.index, data.values)
        plt.title('부서별 연봉 합')
        plt.savefig(img_path, dpi=130)
        plt.close()

        #4. 성별, 직급별 빈도표를 출력하시오.
        img_path2 = static_app_dir / 'myGraph2.png'
        value_counts = mydf['직급'].value_counts()
        print(value_counts)
        plt.figure()
        value_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.title('직급별 빈도표')
        plt.ylabel('')
        plt.axis('equal')
        plt.savefig(img_path2, dpi=130)
        plt.close()

        img_path3 = static_app_dir / 'myGraph3.png'
        value_counts = mydf['성별'].value_counts()
        print(value_counts)
        plt.figure()
        value_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.title('성별 빈도표')
        plt.ylabel('')
        plt.axis('equal')
        plt.savefig(img_path3, dpi=130)
        plt.close()


        if not mydf.empty:
            mydf_html = mydf.to_html(index=False)
        else:
            mydf_html = 'mydf를 못 읽어옴'

        if not mydf_sorted1.empty:
            mydf_sorted1_html = mydf_sorted1.to_html(index=False)
        else:
            mydf_sorted1_html = 'mydf_sorted1 못읽어옴'

        if not mydf_sorted2.empty:
            mydf_sorted2_html = mydf_sorted2.to_html(index=False)
        else:
            mydf_sorted2_html = 'mydf_sorted2 못읽어옴'

        ctx_dict = {
            'mydf_html' : mydf_html,
            'mydf_sorted1_html' : mydf_sorted1_html,
            'mydf_sorted2_html' : mydf_sorted2_html,
            'gadPaySum' : gadPaySum,
            'gadPayMean' : gadPayMean,
            'sdPaySum' : sdPaySum,
            'sdPayMean' : sdPayMean,
            'cdPaySum' : cdPaySum,
            'cdPayMean' : cdPayMean,
            'adPaySum' : adPaySum,
            'adPayMean' : adPayMean,
        }

    return render(request, 'homework.html', ctx_dict)