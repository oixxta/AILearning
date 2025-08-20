from django.shortcuts import render, redirect
import os
from django.conf import settings
import matplotlib
matplotlib.use('Agg')   # 서버에서 시각화 GUI 없이 저장할 때 사용함.
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

from myapp.models import Survey
import pandas as pd
import numpy as np
import scipy.stats as stats

# Create your views here.
def surveyMain(request):
    return render(request, 'index.html')

def surveyView(request):
    return render(request, 'coffee/coffeesurvey.html')


def surveyProcess(request):
    insertDataFunc(request)
    rdata = list(Survey.objects.all().values())
    #print(rdata)
    ctab, results, df = analysisFunc(rdata) #데이터 분석(이원 카이제곱 검정)

    #차트 처리 함수 호출
    staticImgPath = os.path.join(settings.BASE_DIR, 'static', 'images', 'cobar.png')
    saveBrandBarFunc(df, staticImgPath)

    return render(request, 'coffee/result.html', {
        'ctab': ctab.to_html(),
        'result' : results,
        'df' : df.to_html(index=False)
    })

def analysisFunc(data):
    #귀무가설 : 성별에 따라 선호하는 커피브랜드에 차이가 없음.
    #대립가설 : 성별에 따라 선호하는 커피브랜드에 차이가 있음.
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(), 'No Data detected!', pd.DataFrame()
    
    df = df.dropna(subset=['gender', 'co_survey'])  #결측치 제거
    #범주형 데이터의 숫자화
    df['genNum'] = df['gender'].apply(lambda g:1 if g == '남' else 2)   #dummy 변수
    df['coNum'] = df['co_survey'].apply(lambda c:
        1 if c == '스타벅스' 
        else 2 if c == '커피빈' 
        else 3 if c == '이디아' 
        else 4)
    #print(df)
    ctab = pd.crosstab(index=df['gender'], columns=df['co_survey'])
    #print(ctab)
    #표본 부족 시, 메시지 전달
    #if ctab.size == 0 or ctab.shape[0] < 2 or ctab.shape[1] < 5:
    #    results = "표본이 부족합니다! 카이제곱 검정 수행 불가!"
    #    return ctab, results, df
    #정상적인 카이제곱 검정 가능 시 :
    alpha = 0.05    #유의수준
    st, p, dof, expected = stats.chi2_contingency(ctab)

    #기대 빈도 최소값 체크
    min_expected = expected.min()
    expectedNote = ""
    if min_expected < 5:
        expectedNote = f"<br><small>주의 : 기대 빈도의 최소값이 {min_expected:.2f}로, 5 미만이 있어 카이제곱 가정에 다소 취약함.</small>"
    if p >= alpha:
        results = (
            f"p 값이 {p:.5f}이므로, {alpha}보다 크기 때문에 -> "
            f"성별에 따른 커피브랜드 선호 차이가 없음 (귀무가설 채택)"
        )
    else:
        results = (
            f"p 값이 {p:.5f}이므로, {alpha}보다 작기 때문에 -> "
            f"성별에 따른 커피브랜드 선호 차이가 있음 (대립가설 채택)"
        )
    return ctab, results, df

def insertDataFunc(request):
    if(request.method == 'POST'):
        #print(request.POST.get('gender'), request.POST.get('age'), request.POST.get('co_survey'))
        Survey.objects.create(  #DB에 insert가 됨.
            gender = request.POST.get('gender'),
            age = request.POST.get('age'),
            co_survey = request.POST.get('co_survey'),
        )

def surveyShow(request):
    rdata = list(Survey.objects.all().values())
    #print(rdata)
    ctab, results, df = analysisFunc(rdata) #데이터 분석(이원 카이제곱 검정)

    #차트 처리 함수 호출
    staticImgPath = os.path.join(settings.BASE_DIR, 'static', 'images', 'cobar.png')
    saveBrandBarFunc(df, staticImgPath)

    return render(request, 'coffee/result.html', {
        'ctab': ctab.to_html(),
        'result' : results,
        'df' : df.to_html(index=False)
    })

def saveBrandBarFunc(df, staticImgPath):
    # 브랜드명(x축)
    if df is None or df.empty or 'co_survey' not in df.columns:
        try:
            if os.path.exists(staticImgPath):
                os.remove(staticImgPath)
        except Exception:
            pass
        return False
    
    order = ['스타벅스', '커피빈', '이디아', '탐앤탐스']
    brand_counts = df['co_survey'].value_counts().reindex(order, fill_value = 0)

    #color은 무지개 색
    cmap = plt.get_cmap('rainbow')
    n = len(brand_counts)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    fig = plt.figure()
    ax = brand_counts.plot(kind='bar', width=0.6, color=colors, edgecolor='black')
    ax.set_xlabel('커피 브랜드')
    ax.set_ylabel('선호 건수')
    ax.set_title('커피 브랜드 선호 건수')
    ax.set_xticklabels(order, rotation=0)
    fig.tight_layout()
    fig.savefig(staticImgPath, dpi=120, bbox_inches='tight')
    plt.close(fig)  #메모리 낭비 방지