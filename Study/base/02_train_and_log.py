"""2. 모델 생성 및 저장"""
import mlflow
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# [1] Tracking Server 연결 + 실험 지정 (기록은 이 실험 안의 Run으로 남음)
mlflow.set_tracking_uri("http://10.140.150.81:30500")
mlflow.set_experiment("실험실")

# [2] 데이터 준비 — Iris 데이터셋 로드 후 학습/테스트 분리
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# [3] 학습할 모델의 하이퍼파라미터 정의
params = {"solver": "lbfgs", "max_iter": 1000, "random_state": 42}

# [4] Run 시작 — with 블록이 끝나면 Run이 자동으로 종료/기록 확정됨
with mlflow.start_run():
    # [5] 하이퍼파라미터를 Run에 기록
    mlflow.log_params(params)

    # [6] 모델 인스턴스 생성 & 학습
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # [7] 테스트셋으로 예측 후 정확도 계산
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # [8] 메트릭 기록 — UI의 Metrics 컬럼에 표시됨
    mlflow.log_metric("accuracy", accuracy)

    # [9] 모델 시그니처 추론 — 입력/출력 스키마. 나중에 추론 시 검증에 사용
    signature = infer_signature(X_train, model.predict(X_train))

    # [10] 모델 저장 + Model Registry 등록
    #      registered_model_name 지정 시 Registry에 'iris-classifier' 이름으로 버전 생성
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_train,
        registered_model_name="iris-classifier",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model URI: {model_info.model_uri}")
