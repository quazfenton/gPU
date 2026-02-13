"""
Example: Template-based Job Submission

This example demonstrates how to submit jobs using templates with the orchestrator.
The orchestrator automatically:
1. Validates inputs against the template schema
2. Estimates resource requirements from the template
3. Routes the job to an appropriate backend based on requirements
"""

from notebook_ml_orchestrator.orchestrator import Orchestrator
from notebook_ml_orchestrator.core.backends.modal_backend import ModalBackend
from notebook_ml_orchestrator.core.models import BackendType

def main():
    # Initialize orchestrator
    print("Initializing orchestrator...")
    orchestrator = Orchestrator(templates_dir="templates")
    
    # Register a backend (in this example, we'll use Modal)
    print("\nRegistering Modal backend...")
    modal_backend = ModalBackend(
        backend_id="modal-1",
        api_token="your-modal-token-here"  # Replace with actual token
    )
    orchestrator.backend_router.register_backend(modal_backend)
    
    # List available templates
    print("\nAvailable templates:")
    templates = orchestrator.list_templates()
    for template in templates:
        print(f"  - {template.name} ({template.category}): {template.description}")
    
    # Example 1: Submit a speech recognition job
    print("\n" + "="*60)
    print("Example 1: Speech Recognition")
    print("="*60)
    
    try:
        job_id = orchestrator.submit_job(
            template_name="speech-recognition",
            inputs={
                "audio": "path/to/audio.wav",
                "language": "en",
                "model_size": "base"
            },
            user_id="user-123",
            routing_strategy="cost-optimized"
        )
        print(f"✓ Job submitted successfully: {job_id}")
        print(f"  Template: speech-recognition")
        print(f"  Status: Queued and routed to backend")
        
    except Exception as e:
        print(f"✗ Job submission failed: {str(e)}")
    
    # Example 2: Submit an object detection job
    print("\n" + "="*60)
    print("Example 2: Object Detection")
    print("="*60)
    
    try:
        job_id = orchestrator.submit_job(
            template_name="object-detection",
            inputs={
                "image": "path/to/image.jpg",
                "confidence_threshold": 0.7,
                "model": "yolov8n"
            },
            user_id="user-123"
        )
        print(f"✓ Job submitted successfully: {job_id}")
        print(f"  Template: object-detection")
        print(f"  Status: Queued and routed to backend")
        
    except Exception as e:
        print(f"✗ Job submission failed: {str(e)}")
    
    # Example 3: Submit a job with invalid inputs (will fail validation)
    print("\n" + "="*60)
    print("Example 3: Invalid Input Validation")
    print("="*60)
    
    try:
        job_id = orchestrator.submit_job(
            template_name="object-detection",
            inputs={
                # Missing required 'image' field
                "confidence_threshold": 0.7
            },
            user_id="user-123"
        )
        print(f"✓ Job submitted: {job_id}")
        
    except Exception as e:
        print(f"✗ Job submission failed (expected): {type(e).__name__}")
        print(f"  Error: {str(e)}")
    
    # Example 4: Get template metadata
    print("\n" + "="*60)
    print("Example 4: Template Metadata")
    print("="*60)
    
    metadata = orchestrator.get_template_metadata("object-detection")
    if metadata:
        print(f"Template: {metadata['name']}")
        print(f"Category: {metadata['category']}")
        print(f"Description: {metadata['description']}")
        print(f"GPU Required: {metadata['gpu_required']}")
        print(f"Memory: {metadata['memory_mb']} MB")
        print(f"Timeout: {metadata['timeout_sec']} seconds")
        print(f"\nInputs:")
        for inp in metadata['inputs']:
            required = "required" if inp['required'] else "optional"
            print(f"  - {inp['name']} ({inp['type']}, {required}): {inp['description']}")
        print(f"\nOutputs:")
        for out in metadata['outputs']:
            print(f"  - {out['name']} ({out['type']}): {out['description']}")
    
    # Cleanup
    print("\n" + "="*60)
    print("Shutting down orchestrator...")
    orchestrator.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
