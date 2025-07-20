import mlflow
from sentence_transformers import CrossEncoder
from mlflow.tracking import MlflowClient
import os

MODEL_NAME = "CrossEncoderReranker"
SAVE_PATH = "saved_model"


class CrossEncoderWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(os.path.join(context.artifacts["model_path"]))

    def predict(self, context, model_input):
        return self.model.predict(model_input)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")

    # Save HuggingFace CrossEncoder model locally
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model.save(SAVE_PATH)

    # Start MLflow run and log model using pyfunc
    with mlflow.start_run(run_name="Register base CrossEncoder") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CrossEncoderWrapper(),
            artifacts={"model_path": SAVE_PATH}
        )

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        # Transition to Production
        client = MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"Registered CrossEncoder model as Production v{mv.version}")

    # Test stage
    # model = mlflow.pyfunc.load_model("models:/CrossEncoderReranker/Production")
    #
    # model_input = [("what is mlflow?", "mlflow is a tool for experiment tracking")]
    # scores = model.predict(model_input)
    #
    # print(scores)
