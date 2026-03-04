"""Batch data processor - Production Ready."""
import modal
import pandas as pd
import numpy as np
from typing import List, Dict

app = modal.App("batch-processor")
image = modal.Image.debian_slim().pip_install(
    "pandas", "numpy"
)

MAX_BATCH_SIZE = 1000  # Maximum items per batch

@app.function(image=image, timeout=600)
@modal.web_endpoint(method="POST")
def process_batch(data: dict):
    """Process a batch of data items."""
    try:
        # Validate input
        items = data.get('items', [])
        if not isinstance(items, list):
            return {"error": "'items' must be a list"}, 400

        # Enforce batch size limit
        if len(items) > MAX_BATCH_SIZE:
            return {
                "error": f"Batch size exceeds maximum of {MAX_BATCH_SIZE}",
                "received": len(items)
            }, 413

        results = []
        errors = []

        for idx, item in enumerate(items):
            try:
                if not isinstance(item, dict):
                    errors.append({"index": idx, "error": "Item must be a dictionary"})
                    continue

                values = item.get('values', [])
                if not isinstance(values, (list, tuple)):
                    errors.append({"index": idx, "error": "'values' must be a list"})
                    continue

                # Convert to numeric, handling errors
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v) if v is not None else 0)
                    except (ValueError, TypeError):
                        numeric_values.append(0)

                result = {
                    "id": item.get('id', f"item_{idx}"),
                    "mean": float(np.mean(numeric_values)) if numeric_values else 0,
                    "sum": float(np.sum(numeric_values)) if numeric_values else 0,
                    "count": len(numeric_values),
                    "min": float(np.min(numeric_values)) if numeric_values else 0,
                    "max": float(np.max(numeric_values)) if numeric_values else 0,
                }
                results.append(result)

            except Exception as e:
                errors.append({"index": idx, "error": str(e)})

        return {
            "results": results,
            "processed_count": len(results),
            "error_count": len(errors),
            "errors": errors[:100],  # Limit error output
            "total_items": len(items),
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}, 500
