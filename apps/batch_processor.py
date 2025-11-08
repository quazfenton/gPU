"""Batch data processor."""
import modal

app = modal.App("batch-processor")
image = modal.Image.debian_slim().pip_install("pandas", "numpy")

@app.function(image=image)
@modal.web_endpoint(method="POST")
def process_batch(data: dict):
    import pandas as pd
    import numpy as np
    
    items = data.get('items', [])
    results = []
    
    for item in items:
        result = {
            "id": item.get('id'),
            "mean": np.mean(item.get('values', [])),
            "sum": np.sum(item.get('values', []))
        }
        results.append(result)
    
    return {"results": results, "count": len(results)}
