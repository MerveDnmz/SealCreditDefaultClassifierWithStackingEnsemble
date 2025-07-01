import flwr as fl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from flwr.common import Context, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Status, Code

class FlowerXGBClient(fl.client.Client):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        try:
            booster_bytes = self.model.get_booster().save_raw()
        except Exception:
            booster_bytes = b''
        return fl.common.GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[booster_bytes], tensor_type="bytes"),
        )

    def fit(self, ins: FitIns) -> FitRes:
        if ins.parameters.tensors and ins.parameters.tensors[0]:
            self.model._Booster.load_model(ins.parameters.tensors[0])
        self.model.fit(self.X_train, self.y_train)
        booster_bytes = self.model.get_booster().save_raw()
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[booster_bytes], tensor_type="bytes"),
            num_examples=len(self.X_train),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if ins.parameters.tensors and ins.parameters.tensors[0]:
            self.model._Booster.load_model(ins.parameters.tensors[0])
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=0.0,
            num_examples=len(self.X_test),
            metrics={"accuracy": float(acc)},
        )

def main():
    num_samples = 1000
    num_features = 10
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, 2, size=(num_samples,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    num_clients = 3
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    partitions = np.array_split(idx, num_clients)
    client_partitions = [(X_train[part], y_train[part]) for part in partitions]

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        Xc, yc = client_partitions[cid]
        return FlowerXGBClient(Xc, yc, Xc, yc)

    from flwr.client import ClientApp
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation

    client_app = ClientApp(client_fn=client_fn)
    def server_fn(context: Context):
        strategy = FedAvg(
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients
        )
        config = ServerConfig(num_rounds=10)
        return ServerAppComponents(strategy=strategy, config=config)
    server_app = ServerApp(server_fn=server_fn)

    run_simulation(
        client_app=client_app,
        server_app=server_app,
        num_supernodes=num_clients,
    )

    # Merkezi test
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n--- Merkezi XGBoost Test Sonuçları ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main() 