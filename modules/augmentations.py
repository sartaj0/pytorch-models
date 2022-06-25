import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



trainTranforms =  A.Compose([
	A.Resize(416, 416),
	A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
	A.HorizontalFlip(p=0.5),

	A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,), 
	ToTensorV2()
],bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),)

testTranforms =  A.Compose([
	A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,), 
	ToTensorV2()
])