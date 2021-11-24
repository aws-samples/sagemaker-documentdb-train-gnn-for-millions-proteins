# sagemaker-documentdb-train-gnn-for-millions-proteins
Codes for AWS ML blog post entitled "Training graph neural nets for millions of proteins on Amazon SageMaker and DocumentDB (with MongoDB compatibility)"

## Instructions
### 1. Creating resources
We have prepared the following [AWS CloudFormation template](./cloudformation.yaml) to create the required AWS resources for this post. For instructions on creating a CloudFormation stack, see the video [Simplify your Infrastructure Management using AWS CloudFormation](https://www.youtube.com/watch?v=1h-GPXQrLZw&feature=youtu.be&t=153&app=desktop).
The CloudFormation stack provisions the following:
- A VPC with three private subnets for DocumentDB and two public subnets intended for SageMaker Notebook instance and ML training containers, respectively.
- An Amazon DocumentDB cluster with three nodes, one in each private subnet.
- An AWS Secrets Manager secret to store login credentials for Amazon DocumentDB. This allows us to avoid storing plaintext credentials in our SageMaker instance.
- A SageMaker Notebook instance to prepare data, orchestrate training jobs, and run interactive analyses.

In creating the CloudFormation stack, you need to specify the following:
- Name for your CloudFormation stack
- Amazon DocumentDB username and password (to be stored in Secrets Manager)
- Amazon DocumentDB instance type (default db.r5.large)
- SageMaker instance type (default ml.t3.xlarge)
It should take about 15 minutes to create the CloudFormation stack.

### 2. Prepare protein structures and properties and ingest the data into DocumentDB
The notebook [Prepare_data.ipynb](./Prepare_data.ipynb) handles data preprocessing and ingestion to DocumentDB. This notebook runs in the `conda_pytorch_latest_p36` environment that comes with Sagemaker. If you prefer setting up your own python environment, you can do so by: 

```bash
# set up virtual environment:
$ python3 -m venv venv
$ source venv/bin/activate
# install dependencies:
$ pip install -r requirements.txt
```
### 3. Train a GNN on the protein structures using SageMaker
The notebook [Train_and_eval.ipynb] first trains a GNN model on the protein structure datasets stored in the DocumentDB; then loads and evaluates the trained GNN model. 

### 4. Cleaning up
To avoid incurring future charges, delete the CloudFormation stack you created. This removes all the resources you provisioned using the CloudFormation template, including the VPC, Amazon DocumentDB cluster, and SageMaker instance. For instructions, see [Deleting a stack on the AWS CloudFormation console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-delete-stack.html).
