"""
Neural Style Transfer Template for applying artistic styles to images.

This template uses neural style transfer techniques to combine the content
of one image with the artistic style of another, producing a stylized output
image using pre-trained VGG feature extraction and iterative optimization.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class StyleTransferTemplate(Template):
    """
    Applies artistic style transfer between images.
    
    Uses a neural style transfer approach based on VGG19 feature extraction
    to blend the content of a content image with the style of a style image.
    The optimization balances content preservation and style application via
    configurable weight parameters.
    
    Returns the stylized output image.
    """
    
    name = "style-transfer"
    category = "Vision"
    description = "Apply neural style transfer to combine content and style images"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="content_image",
            type="image",
            description="Content image to stylize",
            required=True
        ),
        InputField(
            name="style_image",
            type="image",
            description="Style image to extract artistic style from",
            required=True
        ),
        InputField(
            name="style_weight",
            type="number",
            description="Weight for style loss (higher = stronger style)",
            required=False,
            default=1000000
        ),
        InputField(
            name="content_weight",
            type="number",
            description="Weight for content loss (higher = more content preservation)",
            required=False,
            default=1
        )
    ]
    
    outputs = [
        OutputField(
            name="stylized_image",
            type="image",
            description="Stylized output image"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 600
    pip_packages = ["torch", "torchvision", "Pillow"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the VGG19 feature extraction model.
        
        Loads a pre-trained VGG19 model from torchvision for extracting
        content and style features. Only the feature layers are used.
        """
        import torch
        import torchvision.models as models
        
        # Load pre-trained VGG19 and extract feature layers
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg = vgg.to(self.device)
        
        # Freeze all VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        
        # Define content and style layer indices in VGG19
        self.content_layers = [21]  # conv4_2
        self.style_layers = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        
        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(-1, 1, 1)
        
        self._initialized = True
    
    def _load_image(self, image_input, max_size=512):
        """Load and preprocess an image for style transfer."""
        from PIL import Image
        import torchvision.transforms as transforms
        
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif hasattr(image_input, 'convert'):
            image = image_input.convert('RGB')
        else:
            image = Image.fromarray(image_input).convert('RGB')
        
        # Resize while maintaining aspect ratio
        size = min(max_size, max(image.size))
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
        # Add batch dimension
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def _gram_matrix(self, tensor):
        """Compute the Gram matrix for style representation."""
        b, c, h, w = tensor.size()
        features = tensor.view(b * c, h * w)
        gram = features @ features.t()
        return gram / (c * h * w)
    
    def _get_features(self, image):
        """Extract content and style features from an image using VGG19."""
        features = {}
        x = image
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.content_layers:
                features[f'content_{i}'] = x
            if i in self.style_layers:
                features[f'style_{i}'] = x
        return features
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute neural style transfer on the provided images.
        
        Args:
            content_image: Path to content image or image data
            style_image: Path to style image or image data
            style_weight: Weight for style loss (optional, defaults to 1000000)
            content_weight: Weight for content loss (optional, defaults to 1)
            
        Returns:
            Dict containing:
                - stylized_image: The stylized output image as a PIL Image
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import torch
        from PIL import Image
        
        # Extract parameters
        content_image = kwargs['content_image']
        style_image = kwargs['style_image']
        style_weight = float(kwargs.get('style_weight', 1000000))
        content_weight = float(kwargs.get('content_weight', 1))
        
        # Load and preprocess images
        content_tensor = self._load_image(content_image)
        style_tensor = self._load_image(style_image)
        
        # Normalize images
        content_norm = (content_tensor - self.mean) / self.std
        style_norm = (style_tensor - self.mean) / self.std
        
        # Extract target features
        content_features = self._get_features(content_norm)
        style_features = self._get_features(style_norm)
        
        # Compute style Gram matrices
        style_grams = {
            key: self._gram_matrix(value)
            for key, value in style_features.items()
            if key.startswith('style_')
        }
        
        # Initialize output image as a copy of content image
        target = content_tensor.clone().requires_grad_(True)
        
        # Optimize using L-BFGS
        optimizer = torch.optim.LBFGS([target])
        num_steps = 300
        
        for step in range(num_steps):
            def closure():
                optimizer.zero_grad()
                target.data.clamp_(0, 1)
                target_norm = (target - self.mean) / self.std
                target_features = self._get_features(target_norm)
                
                # Content loss
                c_loss = 0
                for key in content_features:
                    if key.startswith('content_'):
                        c_loss += torch.nn.functional.mse_loss(
                            target_features[key], content_features[key]
                        )
                
                # Style loss
                s_loss = 0
                for key in style_grams:
                    target_gram = self._gram_matrix(target_features[key])
                    s_loss += torch.nn.functional.mse_loss(
                        target_gram, style_grams[key]
                    )
                
                total_loss = content_weight * c_loss + style_weight * s_loss
                total_loss.backward()
                return total_loss
            
            optimizer.step(closure)
        
        # Convert output tensor to PIL Image
        target.data.clamp_(0, 1)
        output_image = target.squeeze(0).cpu().detach()
        output_image = output_image.permute(1, 2, 0).numpy()
        output_image = (output_image * 255).clip(0, 255).astype('uint8')
        stylized = Image.fromarray(output_image)
        
        return {
            'stylized_image': stylized
        }
