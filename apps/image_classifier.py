"""Image classification with ResNet50."""
import modal

app = modal.App("image-classifier")
image = modal.Image.debian_slim().pip_install("torch", "torchvision", "pillow")

@app.function(image=image, gpu="T4")
@modal.web_endpoint(method="POST")
def classify(data: dict):
    import torch, torchvision, base64, io
    from PIL import Image
    
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    
    img_data = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(img_data))
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])
    
    with torch.no_grad():
        output = model(transform(img).unsqueeze(0))
    
    return {"class": output.argmax().item(), "confidence": output.max().item()}
