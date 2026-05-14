import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(size=160):
    return A.Compose([
        A.Resize(size, size),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3,7)),
            A.MotionBlur(blur_limit=(5)),
        ], p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])
    
def get_val_transform(size=160):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])