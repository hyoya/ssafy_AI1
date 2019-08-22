import numpy as np
import csv
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"""
./advertising.csv 에서 데이터를 읽어, X와 Y를 만듭니다.

X는 (200, 3) 의 shape을 가진 2차원 np.array,
Y는 (200,) 의 shape을 가진 1차원 np.array여야 합니다.

X는 TV, Radio, Newspaper column 에 해당하는 데이터를 저장해야 합니다.
Y는 Sales column 에 해당하는 데이터를 저장해야 합니다.
"""

# Req 1-1-1. advertising.csv 데이터 읽고 저장
X = np.zeros((200,3))
Y = np.zeros((200,))

f = open('advertising.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for (idx, line) in enumerate(rdr):
    if idx == 0:
        continue
    X[idx-1][0] = float(line[1])
    X[idx-1][1] = float(line[2])
    X[idx-1][2] = float(line[3])
    Y[idx-1] = float(line[4])

# Req 1-1-2. 학습용 데이터와 테스트용 데이터로 분리합니다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

"""
Req 1-2-1.
LinearRegression()을 사용하여 학습합니다.

이후 학습된 beta값들을 학습된 모델에서 입력 받습니다.

참고 자료:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""

lrmodel = LinearRegression().fit(X_train,Y_train)

# Req 1-2-2. 학습된 가중치 값 저장
beta_0 = lrmodel.coef_[0]
beta_1 = lrmodel.coef_[1]
beta_2 = lrmodel.coef_[2]
beta_3 = lrmodel.intercept_

print("Scikit-learn의 결과물")
print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)


# Req. 1-3-1.
# X_test_pred에 테스트 데이터에 대한 예상 판매량을 모두 구하여 len(y_test) X 1 의 크기를 갖는 열벡터에 저장합니다.
X_test_pred = lrmodel.predict(X_test)

"""
Mean squared error값을 출력합니다.
Variance score값을 출력합니다.

함수를 찾아 사용하여 봅니다.
https://scikit-learn.org/stable/index.html
"""
# Req. 1-3-2. Mean squared error 계산
print("Mean squared error: %.2f" % mean_squared_error(Y_test,X_test_pred))
# Req. 1-3-3. Variance score 계산
print("Variance score: %.2f" % r2_score(Y_test,X_test_pred))

# Req. 1-4-1.
def expected_sales(tv, rd, newspaper, beta_0, beta_1, beta_2, beta_3):
    """
    TV에 tv만큼, radio에 rd만큼, Newspaper에 newspaper 만큼의 광고비를 사용했고,
    트레이닝된 모델의 weight 들이 beta_0, beta_1, beta_2, beta_3 일 때
    예상되는 Sales 값을 출력합니다.
    """
    res = (tv * beta_0) + (rd * beta_1) + (newspaper * beta_2) + beta_3

    return res

## Req. 1-4-2.
## test 데이터에 있는 값을 직접적으로 넣어서 예상 판매량 값을 출력합니다.

print("TV: {}, Radio: {}, Newspaper: {} 판매량: {}".format(
   X_test[3][0],X_test[3][1],X_test[3][2],Y_test[3]))

print("예상 판매량: {}".format(expected_sales(
       float(X_test[3][0]),float(X_test[3][1]),float(X_test[3][2]), beta_0, beta_1, beta_2, beta_3)))

"""
Req. 1-5. pickle로 lrmodel 데이터 저장
파일명: model.clf
"""

with open("model.clf", "wb") as f:
    pickle.dump(lrmodel, f)

# Linear Regression Algorithm Part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.
