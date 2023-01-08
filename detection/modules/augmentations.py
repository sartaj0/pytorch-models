import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

zero = (0, 0, 0)
one = (1, 1, 1)
half = (0.5, 0.5, 0.5)

trainTranforms =  A.Compose([
	A.Resize(416, 416),
	A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.3),
	A.HorizontalFlip(p=0.5),

	A.Normalize(mean=half, std=half, max_pixel_value=255,), 
	ToTensorV2()
],bbox_params=A.BboxParams(format="yolo"))

testTranforms =  A.Compose([
	A.Resize(416, 416),

	A.Normalize(mean=half, std=half, max_pixel_value=255,), 
	ToTensorV2()
],bbox_params=A.BboxParams(format="yolo"))