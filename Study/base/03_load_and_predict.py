"""3. 모델 불러오기 추론"""
import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# [1] Tracking Server 연결 — Registry 조회에도 같은 서버 사용
mlflow.set_tracking_uri("http://10.140.150.81:30500")

# [2] Model Registry에서 모델 로드
#     URI 형식:  models:/<이름>/<버전 또는 latest>
#     pyfunc 로 로드하면 어떤 프레임워크든 동일하게 .predict() 호출 가능
model = mlflow.pyfunc.load_model("models:/iris-classifier/latest")

# [3] 추론용 데이터 준비 — 학습 때와 동일한 split 사용해 재현성 확보
X, y = datasets.load_iris(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [4] 예측 수행
predictions = model.predict(X_test)

# [5] 결과 정리 — 실제값과 예측값을 나란히 보기 좋게 DataFrame으로
result = pd.DataFrame(X_test, columns=datasets.load_iris().feature_names)
result["actual"] = y_test
result["predicted"] = predictions
print(result.head(10))
