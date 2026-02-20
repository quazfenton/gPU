"""
Data Augmentation Template for augmenting images.

This template uses albumentations to apply various image augmentation strategies
for data augmentation in machine learning pipelines. Supports standard, aggressive,
color, and geometric augmentation presets.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class DataAugmentationTemplate(Template):
    """
    Augments images using configurable transformation pipelines.
    
    Uses albumentations to apply augmentation transforms to input images.
    Provides several preset augmentation strategies and generates multiple
    augmented copies of the input image.
    """
    
    name = "data-augmentation"
    category = "Vision"
    description = "Augment images using configurable transformation pipelines"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image file to augment",
            required=True
        ),
        InputField(
            name="augmentation_type",
            type="text",
            description="Augmentation preset to apply",
            required=False,
            default="standard",
            options=["standard", "aggressive", "color", "geometric"]
        ),
        InputField(
            name="num_augmented",
            type="number",
            description="Number of augmented images to generate",
            required=False,
            default=5
        )
    ]
    
    outputs = [
        OutputField(
            name="augmented_images",
            type="json",
            description="List of file paths to augmented images"
        ),
        OutputField(
            name="count",
            type="number",
            description="Number of augmented images generated"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = False
    gpu_type = None
    memory_mb = 1024
    timeout_sec = 120
    pip_packages = ["albumentations", "Pillow", "numpy"]
    
    def setup(self) -> None:
        """
        One-time initialization to configure augmentation pipelines.
        
        Sets up albumentations transform pipelines for each augmentation
        preset type.
        """
        import albumentations as A
        
        self.transforms = {
            'standard': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.GaussNoise(p=0.2),
            ]),
            'aggressive': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ]),
            'color': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.CLAHE(clip_limit=4.0, p=0.3),
                A.ChannelShuffle(p=0.2),
            ]),
            'geometric': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45, p=0.7),
                A.Perspective(scale=(0.05, 0.1), p=0.4),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            ]),
        }
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute data augmentation on the provided image.
        
        Args:
            image: Path to image file or image data
            augmentation_type: Augmentation preset (optional, defaults to 'standard')
            num_augmented: Number of augmented copies to generate (optional, defaults to 5)
            
        Returns:
            Dict containing:
                - augmented_images: List of file paths to generated images
                - count: Number of augmented images generated
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        from PIL import Image
        import numpy as np
        import tempfile
        import os
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        image = kwargs['image']
        augmentation_type = kwargs.get('augmentation_type', 'standard')
        num_augmented = int(kwargs.get('num_augmented', 5))
        
        # Load image if it's a path
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Convert to numpy for albumentations
        img_array = np.array(img)
        
        # Get the transform pipeline
        transform = self.transforms[augmentation_type]
        
        # Generate augmented images
        output_dir = tempfile.mkdtemp(prefix='augmented_')
        augmented_paths = []
        
        for i in range(num_augmented):
            augmented = transform(image=img_array)
            aug_image = Image.fromarray(augmented['image'])
            
            output_path = os.path.join(output_dir, f'augmented_{i}.png')
            aug_image.save(output_path)
            augmented_paths.append(output_path)
        
        return {
            'augmented_images': augmented_paths,
            'count': len(augmented_paths)
        }
