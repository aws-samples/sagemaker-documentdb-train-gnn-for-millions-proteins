# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for model training """

# training set ups:
seed = 0

# training defaults:
batch_size = 128
lr = 1e-3
n_epochs = 10
patience = 5

# data preprocessing:
projection = {"y": "$is_AF"}
knn = 3
num_workers = 8
buffer_size = 256

# GNN model:
h_feats = 16
n_classes = 1
