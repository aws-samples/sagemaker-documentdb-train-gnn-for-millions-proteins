import json
import tarfile
from io import BytesIO
import boto3
import torch


def get_secret(stack_name):
    """Get DocumentDB credentials stored in Secrets Manager"""

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager", region_name=session.region_name
    )

    secret_name = f"{stack_name}-DocDBSecret"
    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    secret = get_secret_value_response["SecretString"]

    return json.loads(secret)


def load_sagemaker_model_artifact(s3_bucket, key):
    """Load a PyTorch model artifact (model.tar.gz) produced by a SageMaker
    Training job.
    Args:
        s3_bucket: str, s3 bucket name (s3://bucket_name)
        key: object key: path to model.tar.gz from within the bucket
    Returns:
        state_dict: dict representing the PyTorch checkpoint
    """
    # load the s3 object
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    # read into memory
    model_artifact = BytesIO(obj["Body"].read())
    # parse out the state dict from the tar.gz file
    tar = tarfile.open(fileobj=model_artifact)
    for member in tar.getmembers():
        pth = tar.extractfile(member).read()

    state_dict = torch.load(BytesIO(pth), map_location=torch.device("cpu"))
    return state_dict
