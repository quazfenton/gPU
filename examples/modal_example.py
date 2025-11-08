"""
Example notebook for Modal deployment with GPU support.
Deploy with: python3 runna.py deploy-modal ./examples --gpu A10G
"""

import numpy as np
import torch

# Simple ML model
class SimpleModel:
    def __init__(self):
        self.weights = np.random.randn(10)
    
    def predict(self, features):
        return np.dot(features, self.weights)

# Initialize model
model = SimpleModel()

def process_request(data):
    """
    Main function called by Modal endpoint.
    
    Args:
        data: dict with 'features' key
    
    Returns:
        dict with 'prediction' key
    """
    features = np.array(data.get('features', []))
    
    if len(features) != 10:
        return {'error': 'Expected 10 features'}, 400
    
    prediction = model.predict(features)
    
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    
    return {
        'prediction': float(prediction),
        'gpu_available': gpu_available,
        'gpu_count': torch.cuda.device_count() if gpu_available else 0
    }

# Test locally
if __name__ == '__main__':
    test_data = {'features': list(range(10))}
    result = process_request(test_data)
    print(f"Test result: {result}")
