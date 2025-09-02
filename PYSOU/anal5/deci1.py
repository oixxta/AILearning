# 의사결정 나무(DecisionTree) - CART
# 예측 분류 모두 가능하나 분류가 주 목적임.
# 비모수검정 : 선형성, 정규성, 등분산성 가정이 필요 없음. 단순함에도 불구하고 알고리즘에 비해 성능이 우수함.
# 단점으로는 유의수준 판단 기준이 없음. 또한, 과적합으로 예측 정확도가 낮을 수 있음.


# 키와 머리카락 길이로 성별 구분모델 작성하기.
import matplotlib.pyplot as plt
from sklearn import tree

x = [[180, 15], [177, 42], [156, 35], [174, 65], [161, 28], [160, 5], [170, 12], [176, 75], [170, 22], [175, 28]]    #연속형 데이터(키, 머리카락 길이)
y = ['man', 'woman', 'woman', 'man', 'woman', 'woman', 'man', 'man', 'man', 'woman']

feature_names = ['height', 'hair length']
class_names = ['man', 'woman']

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0) 
#criterion: 분할 지표, 지니 혹은 엔트로피(기본값: 지니, 'genie')
#max_depth: 트리의 최대 깊이를 지정
#min_samples_split: 
#min_samples_leaf:
#min_weight_fraction_leaf:
#max_features: 최저긔 분할을 위해 고려할 최대 feature 수

model.fit(x, y) #모델 학습
print('훈련 데이터 정확도 : {:.3f}'.format(model.score(x, y)))  #훈련 데이터 정확도 : 0.900
print('예측 결과 : ', model.predict(x)) #['man' 'man'   'woman' 'man' 'woman' 'woman' 'man' 'man' 'man' 'woman']
print('실제값 : ', y)                   #['man' 'woman' 'woman' 'man' 'woman' 'woman' 'man' 'man' 'man' 'woman']


# 새로운 자료로 위의 모델을 사용해 분류하기
newData = [[199, 60]]
print('예측 결과 : ', model.predict(newData))   #['man'], 해당 모델이 키를 성별판단에 더 중요하게 보고 있음.

# 시각화 하기
plt.figure(figsize=(10, 6))
tree.plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=12)
plt.show()  #엔트로피(순수도)가 0이 될 때까지 클래스를 계속 나눔.
