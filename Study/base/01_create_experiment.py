"""1. 실험실 생성"""
import mlflow

# [1] Tracking Server 주소 지정 — 클라이언트가 어디로 기록을 보낼지
mlflow.set_tracking_uri("http://10.140.150.81:30500")

# [2] 실험(Experiment) 설정 — 같은 이름이 없으면 새로 생성, 있으면 그것을 활성화
mlflow.set_experiment("실험실")

# [3] 확인 출력
print("Experiment ready: 실험실")
