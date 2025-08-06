#넘파이 로그 변환
# 데이터 값들의 편차가 클 경우 사용(스케일링), 
# 편차가 큰 데이터들을 일정 범위 내에 넣어서 편차를 줄이는 데에 사용함.
# (분포 개선), 궁극적인 목적은 모델이 보다 안정적으로 학습할 수 있도록 만들어 주는 장점이 있음.

import numpy as np
np.set_printoptions(suppress=True, precision=6)

def test():
    values = np.array([3.45, 34.5, 0.345, 0.01, 0.1, 10, 100, 1000])    #범위가 다양한 1차원배열
    print(np.log2(3.45), np.log10(3.45), np.log(3.14))    #2진 로그, 10진 로그, 자연로그
    print('원본 자료 :', values)
    log_values = np.log10(values)   #상용로그
    print('log_values : ', log_values)
    ln_values = np.log(values)      #자연로그
    print('ln_values : ', ln_values)

    #표준화(Normalization) : 값을 평균을 기준으로 분포시킴.

    #정규화(Standardization) : 정규화는 데이터의 범위를 0 ~ 1 사이로 변환해 데이터 분포를 조정
    #(데이터 - 최솟값)을 (최댓값 - 최솟값)으로 나눠서 구함

    #로그값의 최소, 최대를 구해 0 ~ 1 사이 범위로 정규화
    min_log = np.min(log_values)    #최솟값
    max_log = np.max(log_values)    #최댓값
    normalized = (log_values - min_log) / (max_log - min_log)
    print('정규화 결과 : ', normalized)     #데이터가 원본데이터와 비교해서 평탄화 되었음.

def log_inverse():
    offset = 1
    log_value = np.log(10 + offset)
    print('log_value : ', log_value)
    original = np.exp(log_value) - offset   #np.exp() : 로그 변환에 역변환 가능
    print('original : ', original)

# 로그 변환용 클래스
class LogTrans:
    def __init__(self, offset:float = 1.0):
        self.offset = offset

    #로그 변환 수행 메서드
    def transform(self, x:np.ndarray):
        # fx() = log(x + offset)
        return np.log(x + self.offset)
    
    #역변환(원래 값으로 복원) 수행 메서드
    def inverseTrans(self, x_log:np.ndarray):
        return np.exp(x_log) - self.offset


def gogo():
    print('~' * 20)
    data = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], dtype=float)
    #로그 변환용 클래스 객체 생성
    Log_Trans = LogTrans(offset=1.0)

    #데이터를 로그변환 및 역변환 수행
    data_log_scaled = Log_Trans.transform(data)
    recover_data = Log_Trans.inverseTrans(data_log_scaled)

    print('원본 데이터 : ', data )
    print('로그변환된 데이터 : ', data_log_scaled)
    print('역변환된 데이터 : ', recover_data)


if __name__ == "__main__":
    test()
    log_inverse()
    gogo()

