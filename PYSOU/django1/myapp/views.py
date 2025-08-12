from django.shortcuts import render
from django.conf import settings
from pathlib import Path
import seaborn as sns
import matplotlib
matplotlib.use('Agg')   #matplotlib가 그래프를 그릴 때 backend로 지정하는 코드
#기본은 그래프 창이 열리는데, Agg, PDF -> GUI가 없이 시각화를 파일로 저장함.
import matplotlib.pyplot as plt

# Create your views here.
def main(request):
    return render(request, 'main.html')

def showData(request):
    df = sns.load_dataset('iris')   #iris 데이터프레임을 로드
    #이미지 저장 경로 설정 < BASE_DIR/static/images/iris.png 으로
    static_app_dir = Path(settings.BASE_DIR) / 'static' / 'images'
    static_app_dir.mkdir(parents=True, exist_ok=True)
    img_path = static_app_dir / 'iris.png'

    #파이 차트 저장
    counts = df['species'].value_counts().sort_index() #꽃의 건수 저장
    print('counts : ', counts)  #개발자 디버깅용
    plt.figure()                #그래프 크기 : 기본크기
    counts.plot.pie(autopct = '%1.1f%%', startangle = 90, ylabel='') #파이차트로 그리기, 소수 첫째자리까지만 표현
    plt.title('iris species count')               #그래프 이름 지정
    plt.axis('equal')                             #x축과 y축 길이를 같게 하기
    plt.tight_layout()
    plt.savefig(img_path, dpi=130)                #이미지 저장경로, 해상도 1인치당 130개의 픽셀로.
    plt.close()                                   #메모리 누수 방지용 종료

    #데이터프레임을 테이블 테그로 만들어서 show.html에 전달
    table_html = df.to_html(classes = 'tabel table-striped table-sm', index=False)

    return render(request, 'show.html', {
        'table': table_html,
        'img_relpath' : 'images/iris.png',
    })



