import albumentations as A
from albumentations.pytorch import ToTensorV2
    
def get_val_transform(size=160):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
