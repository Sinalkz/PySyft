import syft as sy                           # Federated Learning Library
from ucimlrepo import fetch_ucirepo         # For Dataset Download 
import numpy as np                          # Mock Dataset Generation

data_site = sy.orchestra.launch(name="cancer-research-center", reset=True)

client = data_site.login(email="info@openmined.org", password="changethis")

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Metadata
metadata = breast_cancer_wisconsin_diagnostic.metadata
# Variable information
variables = breast_cancer_wisconsin_diagnostic.variables

SEED = 12345
np.random.seed(SEED)

X_mock = X.apply(lambda s: s + np.mean(s) + np.random.uniform(size=len(s)))
y_mock = y.sample(frac=1, random_state=SEED).reset_index(drop=True)

features_asset = sy.Asset(
    name="Breast Cancer Data: Features",
    data = X,      # real data
    mock = X_mock  # mock data
)

targets_asset = sy.Asset(
    name="Breast Cancer Data: Targets",
    data = y,      # real data
    mock = y_mock  # mock data
)

# Metadata
description = f'{metadata["abstract"]}\n{metadata["additional_info"]["summary"]}'

paper = metadata["intro_paper"]
citation = f'{paper["authors"]} - {paper["title"]}, {paper["year"]}'

summary = "The Breast Cancer Wisconsin dataset can be used to predict whether the cancer is benign or malignant."

# Dataset creation
breast_cancer_dataset = sy.Dataset(
    name="Breast Cancer Biomarker",
    description=description,
    summary=summary,
    citation=citation,
    url=metadata["dataset_doi"],
)

breast_cancer_dataset.add_asset(features_asset)

breast_cancer_dataset.add_asset(targets_asset)

breast_cancer_dataset

client.upload_dataset(dataset=breast_cancer_dataset)