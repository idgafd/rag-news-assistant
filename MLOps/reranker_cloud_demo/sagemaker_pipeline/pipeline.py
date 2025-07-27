import os
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor, ProcessingOutput

from dotenv import load_dotenv

load_dotenv()
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession(default_bucket="crossencoder-pipeline-data")
image_uri = "416607071613.dkr.ecr.eu-north-1.amazonaws.com/crossencoder-sagemaker:latest"

bucket = 'crossencoder-pipeline-data'
data_prefix = "crossencoder_pipeline"

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')


def make_step(name, script_name, job_args=None, outputs=None, property_files=None, env_vars=None):
    """Helper function to create ProcessingStep with optional environment variables"""
    processor_env = {
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "PYTHONPATH": "/opt/ml/code"
    }
    if env_vars:
        processor_env.update(env_vars)

    return ProcessingStep(
        name=name,
        processor=ScriptProcessor(
            image_uri=image_uri,
            role=role,
            instance_count=1,
            instance_type="ml.t3.medium",
            command=["python3"],
            sagemaker_session=pipeline_session,
            env=processor_env
        ),
        code=os.path.join("scripts", script_name),
        job_arguments=job_args or [],
        outputs=outputs or [],
        property_files=property_files or []
    )


# Load shift data
load_shift_step = make_step(
    name="LoadShiftData",
    script_name="load_data.py",
    job_args=["--mode", "shift",
              "--output_path", f"s3://{bucket}/{data_prefix}/shift_data",
              "--lookback_days", "7"]
)

# Track data drift
shift_property = PropertyFile(
    name="shift",
    output_name="output",
    path="shift_result.json"
)

track_shift_step = make_step(
    name="TrackDataDrift",
    script_name="track_shift.py",
    job_args=[
        "--input_path", f"s3://{bucket}/{data_prefix}/shift_data",
        "--output_path", "/opt/ml/processing/output/shift_result.json",
        "--comparison_window_days", "7",
        "--drift_threshold", "0.25"
    ],
    outputs=[ProcessingOutput(output_name="output", source="/opt/ml/processing/output", destination=f"s3://{bucket}/{data_prefix}/drift_tracking")],
    property_files=[shift_property]
)

# Load training data
load_train_step = make_step(
    name="LoadTrainData",
    script_name="load_data.py",
    job_args=["--mode", "train", "--output_path", f"s3://{bucket}/{data_prefix}/full_data"]
)

# Prepare datasets
prepare_step = make_step(
    name="PrepareDatasets",
    script_name="prepare.py",
    job_args=[
        "--input_path", f"s3://{bucket}/{data_prefix}/full_data",
        "--train_output", f"s3://{bucket}/{data_prefix}/train",
        "--eval_output", f"s3://{bucket}/{data_prefix}/eval",
        "--test_split_ratio", "0.15",
        "--validation_split_ratio", "0.15",
        "--target_balance", "0.3"
    ]
)

# Train model
train_step = make_step(
    name="TrainModel",
    script_name="train.py",
    job_args=[
        "--train_path", f"s3://{bucket}/{data_prefix}/train",
        "--output_model", f"s3://{bucket}/{data_prefix}/new_model",
        "--num_epochs", "3",
        "--batch_size", "16",
        "--learning_rate", "2e-5",
        "--num_negatives", "4",
        "--mlflow_tracking_uri", MLFLOW_TRACKING_URI,
    ]
)

# Evaluate new model
eval_new_property = PropertyFile(
    name="evaluation_new",
    output_name="output",
    path="evaluation.json"
)

evaluate_new_step = make_step(
    name="EvaluateNewModel",
    script_name="evaluate.py",
    job_args=[
        "--model_type", "new",
        "--model_path", f"s3://{bucket}/{data_prefix}/new_model",
        "--eval_path", f"s3://{bucket}/{data_prefix}/eval",
        "--output", "/opt/ml/processing/output/evaluation.json"
    ],
    outputs=[ProcessingOutput(output_name="output", source="/opt/ml/processing/output", destination=f"s3://{bucket}/{data_prefix}/evaluation")],
    property_files=[eval_new_property]
)

# Evaluate production model
eval_prod_property = PropertyFile(
    name="evaluation_prod",
    output_name="output",
    path="evaluation.json"
)

evaluate_prod_step = make_step(
    name="EvaluateProductionModel",
    script_name="evaluate.py",
    job_args=[
        "--model_type", "production",
        "--model_path", f"s3://{bucket}/{data_prefix}/prod_model",
        "--eval_path", f"s3://{bucket}/{data_prefix}/eval",
        "--output", "/opt/ml/processing/output/evaluation.json"
    ],
    outputs=[ProcessingOutput(output_name="output", source="/opt/ml/processing/output", destination=f"s3://{bucket}/{data_prefix}/evaluation")],
    property_files=[eval_prod_property]
)

# Compare metrics
compare_step = ConditionStep(
    name="CompareModels",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step=evaluate_new_step, property_file=eval_new_property, json_path="ndcg"),
            right=JsonGet(step=evaluate_prod_step, property_file=eval_prod_property, json_path="ndcg")
        )
    ],
    if_steps=[
        make_step(
            name="DeployNewModel",
            script_name="deploy.py",
            job_args=["--model_path", f"s3://{bucket}/{data_prefix}/new_model"]
        )
    ],
    else_steps=[]
)

# Shift check condition
shift_condition_step = ConditionStep(
    name="CheckShiftTrigger",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step=track_shift_step, property_file=shift_property, json_path="shift_score"),
            right=0.25
        )
    ],
    if_steps=[
        load_train_step,
        prepare_step,
        train_step,
        evaluate_new_step,
        evaluate_prod_step,
        compare_step
    ],
    else_steps=[]
)

# Set step dependencies
track_shift_step.add_depends_on([load_shift_step])
prepare_step.add_depends_on([load_train_step])
train_step.add_depends_on([prepare_step])
evaluate_prod_step.add_depends_on([prepare_step])
evaluate_new_step.add_depends_on([train_step])
compare_step.add_depends_on([evaluate_new_step, evaluate_prod_step])


pipeline = Pipeline(
    name="CrossEncoderChampionChallengerPipeline",
    steps=[
        load_shift_step,
        track_shift_step,
        shift_condition_step
    ],
    sagemaker_session=pipeline_session
)


pipeline.upsert(role_arn=role)
execution = pipeline.start()
execution.describe()
