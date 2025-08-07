# matplotlib는 플로팅 모듈. 다양한 시각화 그래프 함수 지원.
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.

"""
x = ['seoul', 'incheon', 'suwon']   #x축 데이터, 리스트와 튜플 지원, 셋은 안됨.(인덱스가 없기 때문)
y = [5, 3, 7]   #y축 데이터
plt.xlim([-1, 3])   #x축 경계값을 임의로 선택 가능, -1부터 3까지
plt.ylim([0, 10])   #x축 경계값을 임의로 선택 가능, 0부터 10까지
plt.plot(x, y)          #그래프 만들기
plt.yticks(list(range(0, 10, 3)))
plt.show()              #그래프 보기, 창을 띄우고 아래의 코드 실행을 일시정지함.
#그래프 내용물은 각각 (0,5), (1,3), (2,7)로 저장됨.
#jupiter notebook에선 '%matplotlib inline'을 하면 show() 없어도 됨.
print('ok')             #그래프 창이 꺼져야 실행.


data = np.arange(1, 11, 2)
print(data) #[1 3 5 7 9], 구간 4
plt.plot(data)  #y축 값으로 자동 지정됨.
x = [0, 1, 2, 3, 4]
for a, b in zip(x, data):
    plt.text(a, b, str(b))
plt.show()

plt.plot(data)
plt.plot(data, data, 'r')   # r : 선의 색상
for a, b in zip(data, data):
    plt.text(a, b, str(b))
plt.show()


# sin 곡선 그리기
x = np.arange(10)
y = np.sin(x)
print(x, y)
#plt.plot(x, y)
#plt.plot(x, y, 'bo')    #스타일 지정 가능, 파란색(b)+동그라미로 점 표시(o)
#plt.plot(x, y, 'r+')    #스타일 지정 가능, 빨간색(r)+십자가로 점 표시(+)
plt.plot(x, y, 'go--', linewidth=2, markersize=12)  # - : 솔리드 라인, -- : 대쉬드라인 ...
plt.show()


# 홀드 명령 : 하나의 영역에 두 개 이상의 그래프를 그리기
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.figure(figsize=(10, 5)) #그래프 전체에 대한 너비와 높이 지정 가능
plt.plot(x, y_sin, 'r') #선으로 그리기
plt.scatter(x, y_cos)   #산정도로 그리기
plt.xlabel('x축')   #x축 이름 지정
plt.ylabel('y축')   #y축 이름 지정
plt.title('제목')   #차트의 이름 넣기
plt.legend(['sine', 'cosine'])  #범례
plt.show()


# 서브 플롯 : 여러개의 피규어를 한번에 표시
plt.subplot(2, 1, 1)    #2행 1열 첫번째 플롯
plt.plot(x, y_sin)
plt.title('사인')
plt.subplot(2, 1, 2)    #2행 1열 두번째 플롯
plt.plot(x, y_cos)
plt.title('코사인')
plt.show()
"""

#꺽은선 그래프 그리기
name = ['a', 'b', 'c', 'd', 'e']
kor = ['80', '50', '70', '70', '90']
eng = ['60', '70', '80', '70', '60']
plt.plot(name, kor, 'ro-')
plt.plot(name, eng, 'gs-')
plt.ylim([0, 100])
plt.legend(['국어', '영어'], loc='best') #1, 2, 3, 4
plt.grid(True)
fig = plt.gcf()    #이미지 저장 : 추후에 Django에서 호출시키기 위해.
plt.show()
fig.savefig('result.png')

from matplotlib.pyplot import imread #이미지 읽어오기
img = imread('result.png')
plt.imshow(img)
plt.show()