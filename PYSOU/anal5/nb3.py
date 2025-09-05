# 나이브베이즈 분류모델로 텍스트 분류

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
print(data.target_names)
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']

train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])  # 데이터 중 대표항목


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 각 문자열의 콘텐츠를 숫자벡터로 전환

model = make_pipeline(TfidfVectorizer(), MultinomialNB())  # 작업을 연속적으로 진행
model.fit(train.data, train.target)
labels = model.predict(test.data)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mat = confusion_matrix(test.target, labels)  # 오차행렬 보기
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# 하나의 문자열에 대해 예측한 범주 변환용 유틸 함수 작성

def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

print(predict_category('sending a payload to the ISS'))
print(predict_category('discussing islam vs atheism'))
print(predict_category('determining the screen resolution'))