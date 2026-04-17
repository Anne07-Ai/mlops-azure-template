import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "your-subscription-id")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "your-resource-group")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "your-workspace-name")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-classifier")


def get_workspace():
    """
    Connect to Azure ML workspace.

    Returns:
        Azure ML workspace client
    """
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        logger.info(f"Connected to workspace: {WORKSPACE_NAME}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to workspace: {e}")
        raise


def register_model(model_path: str, metrics: dict) -> dict:
    """
    Register a trained model in Azure ML.

    Args:
        model_path: Path to model artifacts
        metrics: Model evaluation metrics

    Returns:
        dict with registration details
    """
    try:
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes

        client = get_workspace()

        model = Model(
            path=model_path,
            name=MODEL_NAME,
            description=f"Churn classifier — accuracy: {metrics.get('accuracy')}",
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                "accuracy": str(metrics.get("accuracy")),
                "f1_score": str(metrics.get("f1_score")),
                "roc_auc": str(metrics.get("roc_auc"))
            }
        )

        registered = client.models.create_or_update(model)
        logger.info(
            f"Model registered: {registered.name} "
            f"version {registered.version}"
        )

        return {
            "name": registered.name,
            "version": registered.version,
            "id": registered.id
        }

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise


def run_training_pipeline(
    compute_target: str = "cpu-cluster",
    experiment_name: str = "churn-training"
) -> dict:
    """
    Run the full training pipeline on Azure ML.

    Args:
        compute_target: Azure ML compute cluster name
        experiment_name: MLflow experiment name

    Returns:
        dict with pipeline run details
    """
    try:
        from azure.ai.ml import command
        from azure.ai.ml.entities import Environment

        client = get_workspace()

        job = command(
            code=".",
            command="python -m model.train",
            environment=Environment(
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                conda_file="conda.yml"
            ),
            compute=compute_target,
            experiment_name=experiment_name,
            display_name="churn-classifier-training"
        )

        returned_job = client.jobs.create_or_update(job)
        logger.info(f"Pipeline job submitted: {returned_job.name}")

        return {
            "job_name": returned_job.name,
            "status": returned_job.status,
            "studio_url": returned_job.studio_url
        }

    except Exception as e:
        logger.error(f"Pipeline run failed: {e}")
        raise


def deploy_model(
    model_name: str = MODEL_NAME,
    endpoint_name: str = "churn-endpoint"
) -> dict:
    """
    Deploy registered model to Azure ML online endpoint.

    Args:
        model_name: Name of registered model
        endpoint_name: Name of online endpoint

    Returns:
        dict with deployment details
    """
    try:
        from azure.ai.ml.entities import (
            ManagedOnlineEndpoint,
            ManagedOnlineDeployment
        )

        client = get_workspace()

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Churn prediction endpoint",
            auth_mode="key"
        )
        client.online_endpoints.begin_create_or_update(
            endpoint
        ).result()

        deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=f"azureml:{model_name}@latest",
            instance_type="Standard_DS2_v2",
            instance_count=1
        )
        client.online_deployments.begin_create_or_update(
            deployment
        ).result()

        logger.info(
            f"Model deployed to endpoint: {endpoint_name}"
        )

        return {
            "endpoint_name": endpoint_name,
            "deployment_name": "blue",
            "status": "deployed"
        }

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise