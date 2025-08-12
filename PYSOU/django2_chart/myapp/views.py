from django.shortcuts import render
import json, os
import pandas as pd
import numpy as np
import requests
from django.conf import settings
from datetime import datetime

# Create your views here.

DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'seattle_weather.csv')
CSV_URL = 'https://raw.githubusercontent.com/vega/vega-datasets/master/data/seattle-weather.csv'

def index(request):
    return render(request, 'index.html')

# csv 데이터 파일이 없으면 다운로드해서 저장하는 역할
def csvFunc():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        res = requests.get(CSV_URL, timeout=20)
        res.raise_for_status()  #http 상태 코드가 200(성공)이 아니면 예외를 발생시킴.

        with open(CSV_PATH, mode = 'wb') as f:
            f.write(res.content)

def show(request):
    csvFunc()   # CSV 데이터 확보
    df = pd.read_csv(CSV_PATH)
    print(df.columns) #['date', 'precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
    print(df.info())

    #일부 열만 작업에 참여시키기
    df = df[['date', 'precipitation', 'temp_max', 'temp_min']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()

    #기술통계 - 평균, 표준편차, 최소값, 최댓값, 4분위수, ...
    stats_df = df[['precipitation', 'temp_max', 'temp_min']].describe().round(3)
    #print('stats_df : ', stats_df)
    
    #df의 상위 5행 출력
    head_html = df.head(5).to_html(classes='table table-sm table-striped', index=False, border=0)
    stats_html = stats_df.to_html(classes='table table-sm table-striped', index=False, border=0)

    # Echart용 데이터(월별 평균 최고 기온)
    # 월 단위 평균 최고기온 집계
    monthly = (
        df.set_index('date').resample('ME')[['temp_max']].mean().reset_index()
    )
    print('monthly : ', monthly.head(2))

    # 2012-01-31 7.054839
    labels = monthly['date'].dt.strftime('%Y-%m').tolist()
    series = monthly['temp_max'].round(2).tolist()


    ctx_dic = {
        'head_html': head_html,
        'stats_html' : stats_html,
        'labels_json' : json.dumps(labels, ensure_ascii=False), #리스트 타입을 제이슨 스트링으로 바꿔줌.
        'series_json' : json.dumps(series, ensure_ascii=False),
    }

    return render(request, 'show.html', ctx_dic)
