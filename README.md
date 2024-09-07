# 데이콘 AI 경진대회3

# 일시
- 2024-01-15 ~ 2024-02-05

# 주제
- 고객 대출등급 분류 AI 해커톤

# 목적
- 주어진 금융 데이터를 활용하여 고객의 대출등급을 예측
 

# 데이터
- https://dacon.io/competitions/official/236214/data   (데이터 출처)
- 데이터 개수 : 96294개
  
| Column          | Non-Null Count | Dtype  |
|-----------------|----------------|--------|
| ID              | 96294 non-null | object |
| 대출금액           | 96294 non-null | int64  |
| 대출기간           | 96294 non-null | object |
| 근로기간           | 96294 non-null | object |
| 주택소유상태         | 96294 non-null | object |
| 연간소득           | 96294 non-null | int64  |
| 부채_대비_소득_비율  | 96294 non-null | float64|
| 총계좌수           | 96294 non-null | int64  |
| 대출목적           | 96294 non-null | object |
| 최근_2년간_연체_횟수 | 96294 non-null | int64  |
| 총상환원금         | 96294 non-null | int64  |
| 총상환이자         | 96294 non-null | float64|
| 총연체금액         | 96294 non-null | float64|
| 연체계좌수         | 96294 non-null | float64|
| 대출등급           | 96294 non-null | object |


  

# 사용언어/ 최종 선정 모델
- python/ stacking(randomforest + decision tree + xgboost + lgbm)

# 모델 성능 지표
- macro f1 score

# EDA
- 대출기간별(30months, 60months) 대출등급 분포 countplot으로 확인(30months의 경우 b등급이 가장 많고 60months의 경우 c등급이 가장 많음)
- 대출기간별 대출금액 displot으로 확인(36months보다 60months일 경우 대출금액이 더 높음)
- 대출기간별 총상환이자 displot으로 확인(36months보다 60months일 경우 총상황이자가 더 높음)
- 주택소유상태별 대출등급 countplot으로 확인(mortage, own인 경우 b등급이 가장 많고 rent인 경우는 c등급이 가장 많음)
- 대출목적별 대출등급 countplot으로 확인(대부분의 대출목적에서 b등급 또는 c등급이 가장 많음)
- 대출목적별 평균 연간 소득, 평균 대출금액, 평균 총 연체금액, barplot으로 확인
- 근로기간별 대출등급 countplot으로 확인(대부분의 근로기간에서 b등급 또는 c등급이 가장 많음)
- 근로기간별 주택소유상태 countplot으로 확인(대부분의 근로기간에서 own 또는 mortage가 가장 많음)
- 근로기간별 평균연간소득 barplot으로 확인(근로기간이 길수록 우상향하는 상황 확인)
- 근로기간별 평균대출금액 barplot으로 확인(근로기간이 길수록 우상향하는 상황 확인)
- 대출등급별 대출금액 확인(대출등급이 좋지 않을수록 대출금액이 높음)
- 대출등급별 월평균 대출금액 barplot으로 확인(a등급이 월 평균 대출금액이 가장 높음)
- 대출등급변 평균 연간소득 barplot으로 확인(a등급이 가장 높음)
- 대출등급별 부채대비 소득비율 barplot으로 확인(e등급이 가장 높음)
- 대출등급별 총계좌수 barplot으로 확인(a등급이 가장 높음)
- 대출등급별 총상환원금 barplot으로 확인(a등급이 가장 높음)
- 대출등급별 총상황이자,총연체금액, 최근 2년간 연체 횟수, 대출금액 총 상환원금 비율 barplot으로 확인(g등급이 가장 높음)
- 대출등급별 연체계좌수 barplot으로 확인(d등급이 가장 높음)
- 대출등급변 대출금액_총상환이자_비율 barplot으로 확인(a 등급이 가장 높음)

  

# 데이터 전처리
- 근로기간,대출기간 변수의 범주들에서 문자열을 제거
- 연간소득 변수의 0값은 대출금액을 부채대비소득비율 변수로 나눈값으로 대체
- 월_대출금액, 월_대출대비_소득비율, 계좌수, 대출금액_총상환이자_비율, 대출금액_총상환원금_비율, 상환이자_상환원금, 총상환액, 소득대비_총상환액_비율, 대출대비_총상환액_비율, 기간대비_총상환액_비율, 대출대비_총상환원금_비율, 대출대비_총상환이자_비율, 소득대비_총상환원금_비율, 소득대비_총상환이자_비율, 기간대비_총상환원금_비율, 기간대비_총상환이자_비율, 월_이자_지불액 변수 feature engineering으로 새로 생성
- "총계좌수", "총상환원금", "총상환이자", "연체계좌수", "총연체금액"변수들은 drop(필요 없거나 많이 사용된 변수라서)
- 대출목적 변수 binary encoding 수행
- 수치형 변수 모두 standard scaler를 통해 표준화 처리
- lda를 통해 나온 중요 변수 2개 독립변수로 추가
- 종속변수 문자열 범주를 라벨 인코딩을 통해 정수형으로 변환


# 모델링
- parameter(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=2024)를 설정한 후 randomforest 모델 fitting(macro f1: 약 0.92)
- parameter( max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=2024)를 설정한 후 decisiontree 모델 fitting(macro f1: 약 0.9076)
- parameter(random_state=2024, n_estimators=1000, learning_rate=0.01, depth=10,l2_leaf_reg=3, metric_period=1000, verbose=1000)를 설정한 후 catboost 모델 fitting(macro f1: 약 0.87)
- decision tree parameter(max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=2024),  xgboost parameter(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=2024), lgbm parameter(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=2024)를 설정하고 stacking model(최종모형 randomforest(n_estimators=100,random_state=2024)) fitting(macro f1: 약 0.95) -> 최종 모델 선정


# 느낀점
- 이 데이터는 수치형 변수 데이터가 많은데 이 수치형 변수들의 다양한 조합으로 feature engineering을 수행하여 다양한 변수를 생성했기 때문에 좋은 결과가 낭로 수 있었다고 생각한다. 다음 대회 때도 다양한 변수들의 조합으로 적절한 feature engineering을 해야겠다는 생각을 했다.
