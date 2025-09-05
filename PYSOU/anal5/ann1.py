"""
미분계수 연습

평균변화율과 순간변화율을 코드로 작성하기
서울과 부산까지 자동차를 차고 이동하는 경우, 총 거리 400km, 총 소요 시간 4시간, 평균속도 시속 100km/h(평균변화율)
중심차분 공식으로 순간속력을 근사

중심차분 : 양 옆 두 지점의 차이를 이용해, 중간 지점의 순간 기울기를 추정하는 법.
(미분은 순간변화율)
"""

import numpy as np
import matplotlib.pyplot as plt
t = np.array([0, 1, 2, 3, 4], dtype=float)                  #시간 데이터
s = np.array([0.0, 80.0, 180.0, 300.0, 400.0], dtype=float) #누적 이동거리

"""
plt.plot(t, s)
plt.xlabel('t, time')
plt.ylabel('s, speed')
plt.grid()
plt.show()
plt.close()
"""

#전체 주행거리
s_tot = s[-1]
s_half = s_tot / 2.0    #중간 지점 = 200km

#평균 변화율 계산
#평균 속도 = 전체거리 변화량 / 전체시간 변화량
t_tot = t[-1] - t[0]            #총 소요시간, t 전체에서 0반째 요소를 뺌.
v_avg = (s[-1] - s[0]) / t_tot  #전체 평균속도

#보간함수 사용
t_mid = np.interp(s_half, s, t)     # interp : 선형 보간함수.
print('t_mid : ', t_mid)            # t_mid :  2.1666666666666665, 실제 주행곡선은 속도의 변화때문에 더 늦게 200km에 도달함.

#시간 간격의 중앙값을 구함. -> 평균적인 샘플 간격을 계산함.
dt_mid = np.median(np.diff(t))
h = dt_mid * 0.5        # 중심 차분에 사용할 작은 간격 = 0.5h
s_plus = np.interp(t_mid + h, t, s)
s_minus = np.interp(t_mid - h, t, s)

#중심차분으로 순간 속도 추정하기(km/h)
v_kph = (s_plus - s_minus) / (2.0 * h)

print(f'총 이동거리 : {s_tot:.1f}km, 총 소요시간 : {t_tot:.1f}hour')
print(f'평균변화율(평균속력) : {v_avg:.1f}km/h')
print(f'중간지점을 지나는 시간 : {t_mid:.2f}시간')
print(f's_plus : {s_plus:.1f}km, s_minus : {s_minus:.1f}km')
print(f'중간 지점의 순간 속도 : {v_kph:.1f}km/h')   #순간속도 = 순간변화율 = 접선의 기울기 = 미분계수
"""
t_mid :  2.1666666666666665
총 이동거리 : 400.0km, 총 소요시간 : 4.0hour
평균변화율(평균속력) : 100.0km/h
중간지점을 지나는 시간 : 2.17시간
s_plus : 260.0km, s_minus : 146.7km
중간 지점의 순간 속도 : 113.3km/h
"""