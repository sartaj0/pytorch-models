import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
trainTranforms =  A.Compose([
	A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,), 
	ToTensorV2()
])

testTranforms =  A.Compose([
	A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,), 
	ToTensorV2()
])