"""
선형회귀 모델식 계산 - 최소제곱법(ols)으로 y = wx + b 형태의 추세식 파라미터 w와 b를 추정
"""

import numpy as np


class MySimpleLinearRegression():
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, x:np.ndarray, y:np.ndarray):
        #ols로 w와 b를 추정
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        self.w = numerator / denominator
        self.b = y_mean - (self.w * x_mean)

    def predict(self, x:np.ndarray):   #예측값 자동 생성기
        return self.w * x + self.b




def main():
    np.random.seed(42)
    # 임의의 성인 남성 10명의 키와 몸무게 자료를 사용.
    x_heights = np.random.normal(175, 5, 10)  #정규분포, 키
    y_weights = np.random.normal(70, 10, 10)

    #최소제곱법을 수행하는 클래서 객체 생성
    model = MySimpleLinearRegression()
    model.fit(x_heights, y_weights)

    #추정된  w와 b 확인해보기
    print('w : ', model.w)  #-0.23090100700107954
    print('b : ', model.b)  #103.0183826888111
    #1차식, y = wx + b 완성.

    #예측값 확인해보기 : 실제 y값들로.
    y_predict = model.predict(x_heights)
    #print(y_predict)
    print('실제 몸무게와 예측 몸무게의 비교 :')
    for i in range(len(x_heights)):
        print(f"키 : {x_heights[i]:.2f}cm, 실제몸무게 : {y_weights[i]:.2f}kg, 예측몸무게 : {y_predict[i]:.2f}kg")

    print('키 199cm의 키를 가진 미지의 남성의 몸무게 예측하기')
    print(model.predict(199))

if __name__ == "__main__":
    main()