import os
import numpy as np 
from PIL import Image, ImageFile

import torch
from torch.utils import data
from torchvision import transforms as T

from .utils import iou_width_heght as iou


def transformations(args):
	transforms = T.Compose([
		T.Resize((args['imageSize'], args['imageSize'])),
		T.ToTensor()
	])

	return transforms
torch.set_printoptions(profile="full")
ImageFile.LOAD_TRUNCATED_IMAGES = True
class YOLOv3Dataset(data.Dataset):
	def __init__(self, dataDir: str, numClasses: int, anchors: list, S: list, transforms, args: dict):
		imgDir = os.path.join(dataDir, "images")

		annotationDir = os.path.join(dataDir, "labels")

		self.annotations = list()
		suffix = (".jpg", ".png", ".jpeg", ".wpeg", ".JPG")
		for f in os.listdir(imgDir):
			annotFile = f.rsplit('.', 1)[0] + ".txt"
			annotFilePath = os.path.join(annotationDir, annotFile)
			imgFilePath = os.path.join(imgDir, f)
			# print(imgFilePath, annotFilePath)
			if (f.endswith(suffix) & os.path.isfile(annotFilePath)):
				
				self.annotations.append([imgFilePath, annotFilePath])
		
		self.anchors = torch.tensor([i for prAnc in anchors for i in prAnc])
		self.anchorsPerScale = anchors
		self.numAnchors = self.anchors.shape[0]
		self.numAnchorPerScale = len(self.anchorsPerScale[0])
		self.numClasses = numClasses
		self.ignoreIOUThresh = 0.5
		self.S = S
		self.imgSize = 416
		self.scaledAnchors = self.anchors / self.imgSize
		self.transforms = transforms
		# self.transforms = transformations(args)
	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, index):
		bboxes = np.roll(np.loadtxt(fname=self.annotations[index][1], delimiter=" ", ndmin=2), 4, axis=1).tolist()
		bboxes = sorted(bboxes, key=lambda x: x[2] * x[3], reverse=True)

		# image = Image.open(self.annotations[index][0]).convert("RGB")
		# image = self.transforms(image)

		image = np.array(Image.open(self.annotations[index][0]).convert("RGB"))
		if self.transforms:
			augmentations = self.transforms(image=image, bboxes=bboxes)
			image = augmentations["image"]
			bboxes = augmentations["bboxes"]
		
		H, W = image.shape[ :2]

		targets = [torch.zeros(self.numAnchorPerScale, S, S, 6) for i, S in enumerate(self.S)]

		# iou_anchors_all = [iou(torch.tensor(box[2:4]), self.anchors / 416)for box in bboxes]
		for box in bboxes:
			iouAnchors = iou(torch.tensor(box[2:4]), self.scaledAnchors)
			anchorIndices = iouAnchors.argsort(descending=True, dim=0)
			x, y, width, height, classLabel = box
			hasAnchor = [False] * self.numAnchorPerScale
			for anchorIdx in anchorIndices:
				scaleIdx = anchorIdx // self.numAnchorPerScale
				anchorOnScale = anchorIdx % self.numAnchorPerScale
				S = self.S[scaleIdx]
				gridX, gridY = S * x, S * y
				
				i, j = int(gridY), int(gridX)
				anchorTaken = targets[scaleIdx][anchorOnScale, i, j, 0]

				if not anchorTaken and not hasAnchor[scaleIdx]:
					targets[scaleIdx][anchorOnScale, i, j, 0] = 1
					xCell, yCell = gridX - j, gridY - i 

					widthCell, heightCell = S * width, S * height
					targets[scaleIdx][anchorOnScale, i, j, 1:] = torch.tensor(
						[xCell, yCell, widthCell, heightCell, int(classLabel)]
					) 
					hasAnchor[scaleIdx] = True
					# print(x, y, gridX, gridY, S, classLabel)
				elif not anchorTaken and iouAnchors[anchorIdx] > self.ignoreIOUThresh:
					targets[scaleIdx][anchorOnScale, i, j, 0] = -1  # ignore prediction
		return image, tuple(targets)



if __name__ == "__main__":
	imgSize = 416
	S = [imgSize//32, imgSize//16, imgSize//8]
	numClasses = 4
	imgDir = r"E:\dataset\detection\sampleData\export"
	anchors = [[[10,13],  [16,30],  [33,23]],  [[30,61],  [62, 45],  [59, 119]],  [[116,90],  [156,198],  [373,326]]][::-1]
	transforms = False
	dataset = YOLOv3Dataset(imgDir, numClasses, anchors, S=S, transforms=transforms)

	# obj = dataset[17][1][i][..., 0]==1.0
	# print(dataset[17][1][i][obj])
	imgId=0
	print(torch.cat([dataset[imgId][1][i][dataset[imgId][1][i][..., 0]==1.0] for i in range(len(dataset[imgId][1]))], dim=0))