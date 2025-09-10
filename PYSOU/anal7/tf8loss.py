"""
선형회귀모델 추세식 계산
"""
import numpy as np

class LinearRegressionTest:
    def __init__(self, learningRate, epochs):   #클래스 생성자
        self.w = None
        self.b = None
        self.learningRate = learningRate
        self.epochs = epochs
    
    def fit(self, x:np.ndarray, y:np.ndarray):      #학습
        # 경사하강법 이용하기(Gradient Descent)으로 w와 b 학습
        # parameter 초기화
        self.w = np.random.uniform(-2, 2)
        self.b = np.random.uniform(-2, 2)
        n = len(x)

        for epoch in range (self.epochs):
            y_pred = self.w * x + self.b        # 예측값
            loss = np.mean((y - y_pred) ** 2)   # 손실()

            dw = (-2 / n) * np.sum(x * (y - y_pred))    # 경사 계산 : 편미분 사용
            db = (-2 / n) * np.sum(y - y_pred)

            self.w -= self.learningRate * dw
            self.b -= self.learningRate * db    # w와 b의 값 갱신해나가기

            #학습 상태 출력
            if(epoch % 10 == 0):   #10번에 1번씩만 출력하기
                print(f'epoch {epoch + 1} / {self.epochs} = Loss : {loss:.5f}, w : {self.w:.5f}, bias : {self.b:.5f}')

    def predict(self, x:np.ndarray):
        return (self.w * x) + self.b


def main():
    np.random.seed(42)
    # feature
    x_heights = np.random.normal(175, 10, 30)
    true_w = 0.7
    true_b = -55
    noise = np.random.normal(0, 5, 30)
    # label
    y_weights = true_w * x_heights + true_b + noise 
    print(x_heights)
    print(y_weights)
    
    #스케일링(Scaling, 표준화)
    x_mean = np.mean(x_heights)
    x_std = np.std(x_heights)
    y_mean = np.mean(y_weights)
    y_std = np.std(y_weights)

    x_heights_scaled = (x_heights - x_mean) / x_std
    y_weights_scaled = (y_weights - y_mean) / y_std

    #모델 학습하기
    model = LinearRegressionTest(learningRate=0.001, epochs=1000)
    model.fit(x_heights_scaled, y_weights_scaled)

    #예측하기
    y_pred_scaled =  model.predict(x_heights_scaled)
    y_pred = (y_pred_scaled * y_std) + y_mean  #스케일된 값을 넣었기 때문에 역변환 필요! : 표준화 전 원래 값으로
    print('y_pred : ', y_pred)

    #모델 성능 확인해보기(MSE, R^2로)
    mse = np.mean((y_weights - y_pred) ** 2)
    ss_tot = np.sum((y_weights - np.mean(y_weights)) ** 2)
    ss_res = np.sum((y_weights - y_pred) ** 2) 
    r2square = 1 - (ss_res / ss_tot)

    print('추정된 기울기(w) : ', model.w)
    print('추정된 b : ', model.b)

    for i in range(len(x_heights)):
        print(f'키 : {x_heights[i]:.2f}cm, 몸무게 실제 : {y_weights[i]:.2f}kg, 예측값 : {y_pred[i]:.2f}kg')
    
    print('MSE (평균제곱오차) : ', mse)
    print('결정계수 : ', r2square)              # 58%의 설명력을 가짐.

if __name__ == '__main__':
    main()