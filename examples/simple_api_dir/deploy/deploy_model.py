import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub
from huggingface_hub import hf_hub_download
from kagglehub.config import get_kaggle_credentials

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def authenticate_kaggle():
    try:
        kagglehub.login()  # Prompts for credentials if not set
        username, _ = get_kaggle_credentials()
        if not username:
            raise ValueError("Kaggle authentication failed")
        logger.info("Kaggle authentication successful")
    except Exception as e:
        logger.error(f"Kaggle authentication error: {str(e)}")
        raise

def handle_kaggle_interactions(dataset="", competition="", notebook=""):
    try:
        authenticate_kaggle()
        api = KaggleApi()
        api.authenticate()

        # List competitions
        if competition:
            logger.info("Listing active Kaggle competitions")
            competitions = api.competitions_list()
            logger.info(f"Active competitions: {[c.title for c in competitions]}")

        # Download competition files
        if competition:
            logger.info(f"Downloading files for competition: {competition}")
            api.competition_download_files(competition, path=".")

        # List datasets
        if dataset:
            logger.info(f"Listing datasets matching: {dataset}")
            datasets = api.dataset_list(search=dataset)
            logger.info(f"Matching datasets: {[d.title for d in datasets]}")

        # Download dataset
        if dataset:
            logger.info(f"Downloading dataset: {dataset}")
            api.dataset_download_files(dataset, path=".", unzip=True)

        # List and pull notebooks
        if notebook:
            logger.info(f"Listing notebooks matching: {notebook}")
            notebooks = api.kernels_list(search=notebook)
            logger.info(f"Matching notebooks: {[k.title for k in notebooks]}")
            api.kernel_pull(notebook, path=".", metadata=True)

    except Exception as e:
        logger.error(f"Error in Kaggle interactions: {str(e)}")
        raise

def download_model(source, model_path, kaggle_dataset="", kaggle_competition="", hf_repo_id="", hf_model_file="model.pth"):
    try:
        if source == "kaggle":
            authenticate_kaggle()
            if kaggle_competition:
                logger.info(f"Downloading model from Kaggle competition: {kaggle_competition}")
                api = KaggleApi()
                api.authenticate()
                api.competition_download_file(kaggle_competition, model_path, path=".")
            elif kaggle_dataset:
                logger.info(f"Downloading model from Kaggle dataset: {kaggle_dataset}")
                path = kagglehub.model_download(f"{kaggle_dataset}/pyTorch/2b")
                os.rename(os.path.join(path, "model.pth"), model_path)
            else:
                raise ValueError("Kaggle dataset or competition must be specified")
        
        elif source == "huggingface":
            logger.info(f"Downloading model from Hugging Face: {hf_repo_id}")
            hf_hub_download(repo_id=hf_repo_id, filename=hf_model_file, local_dir=".", local_dir_use_symlinks=False)
            os.rename(hf_model_file, model_path)
        
        else:
            raise ValueError(f"Unsupported model source: {source}")
        
        logger.info(f"Model downloaded to {model_path}")
    
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

def load_model(model_path, input_dim, output_dim):
    try:
        model = StudentModel(input_dim, output_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def preprocess_data(data):
    try:
        data_array = np.array(data, dtype=np.float32)
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(data_array.reshape(-1, data_array.shape[-1])).reshape(data_array.shape)
        processed_data = torch.tensor(processed_data, dtype=torch.float32)
        logger.info("Data preprocessed successfully")
        return processed_data
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def predict(model, data):
    try:
        with torch.no_grad():
            prediction = model(data)
        return prediction.numpy()
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise