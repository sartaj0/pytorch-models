import torch 
import torch.nn as nn
import torch.nn.functional as F 

from torch import optim
from torch.utils import data
from torchvision import transforms as T

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary

from models.resenetYolo import ResNetYOLO

class ConvBatchLeakyReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super(ConvBatchLeakyReLU, self).__init__()
		self.convBatch = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
			nn.BatchNorm2d(num_features=out_channels),
			nn.LeakyReLU(0.1)
		)
	def forward(self, x):
		return self.convBatch(x)

class DoubleConvBatch(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels):
		super(DoubleConvBatch, self).__init__()
		self.conv = nn.Sequential(
			ConvBatchLeakyReLU(in_channels, mid_channels, 1, 1, 0),
			ConvBatchLeakyReLU(mid_channels, out_channels, 3, 1, 1)
		)

	def forward(self, x):
		return self.conv(x)

class DetectionLayer(nn.Module):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors

	def forward(self, x):
		return x


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels):
		super(ResidualBlock, self).__init__()

		self.doubleConv = DoubleConvBatch(in_channels, mid_channels, out_channels)
	
	def forward(self, x):
		x = torch.add(self.doubleConv(x), x)
		return x

class OutputLayer(nn.Module):
	def __init__(self, in_channels, mid_channels, anchors, numClasses, anchorPerScale):
		super(OutputLayer, self).__init__()
		self.numClasses = numClasses
		self.anchorPerScale = anchorPerScale
		filter_size = self.anchorPerScale * (self.numClasses + 5)
		self.outputconv = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, stride=1),
			nn.Conv2d(in_channels=mid_channels, out_channels=filter_size, kernel_size=1, padding=0, stride=1),
			DetectionLayer(anchors)
		)
	
	def forward(self, x):
		x = self.outputconv(x).reshape(-1, self.anchorPerScale, (self.numClasses + 5), x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
		return x

class Model(nn.Module):
	def __init__(self, anchors, numClasses):
		super(Model, self).__init__()
		self.numClasses = numClasses
		self.anchorPerScale = 3
		self.filter_size = self.anchorPerScale * (self.numClasses + 5)
		self.anchors = anchors
		self.conv1 = ConvBatchLeakyReLU(in_channels=3, out_channels=32, kernel_size=3, padding=1)
		self.upsampleMode = "nearest"

		self.convres1 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=64, mid_channels=32, out_channels=64) for i in range(1)]
		)

		self.convres2 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=128, mid_channels=64, out_channels=128) for i in range(2)]
		)

		self.convres3 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=256, mid_channels=128, out_channels=256) for i in range(4)],
		)

		self.convres4 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=512, mid_channels=256, out_channels=512) for i in range(4)],
		)

		self.convres5 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=1024, mid_channels=512, out_channels=1024) for i in range(4)]
		)

		self.dcc1 = nn.Sequential(
			DoubleConvBatch(in_channels=1024, mid_channels=512, out_channels=1024),
			DoubleConvBatch(in_channels=1024, mid_channels=512, out_channels=1024),
			ConvBatchLeakyReLU(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		)

		self.output1 = OutputLayer(in_channels=512, mid_channels=1024, anchors=self.anchors[0], 
		numClasses=self.numClasses, anchorPerScale=self.anchorPerScale)

		self.conv7 = ConvBatchLeakyReLU(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.pool1 = nn.Upsample(scale_factor=2, mode=self.upsampleMode)

		self.dcc2 = nn.Sequential(
			DoubleConvBatch(in_channels=768, mid_channels=256, out_channels=512),
			DoubleConvBatch(in_channels=512, mid_channels=256, out_channels=512),
			ConvBatchLeakyReLU(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1)
		)

		self.output2 = OutputLayer(in_channels=256, mid_channels=512, anchors=self.anchors[1], 
		numClasses=self.numClasses, anchorPerScale=self.anchorPerScale)

		self.conv8 = ConvBatchLeakyReLU(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1)
		self.pool2 = nn.Upsample(scale_factor=2, mode=self.upsampleMode)

		self.dcc3 = nn.Sequential(
			DoubleConvBatch(in_channels=384, mid_channels=128, out_channels=256),
			DoubleConvBatch(in_channels=256, mid_channels=128, out_channels=256),
			ConvBatchLeakyReLU(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1)
		)
		self.output3 = OutputLayer(in_channels=128, mid_channels=256, anchors=self.anchors[2], 
		numClasses=self.numClasses, anchorPerScale=self.anchorPerScale)


	def forward(self, x):
		x = self.conv1(x)
		x = self.convres1(x)
		x = self.convres2(x)
		x1 = self.convres3(x)
		x2 = self.convres4(x1)

		x = self.convres5(x2)

		x3 = self.dcc1(x)
		output1 = self.output1(x3)

		x = self.conv7(x3)
		x = self.pool1(x)
		x = torch.cat((x, x2), 1)

		x4 = self.dcc2(x)
		output2 = self.output2(x4)

		x = self.conv8(x4)
		x = self.pool2(x)
		x =  torch.cat((x, x1), 1)

		x = self.dcc3(x)
		output3 = self.output3(x)

		return output1, output2, output3

def transformations(args):
	transforms = T.Compose([
		T.Resize((args['imageSize'], args['imageSize'])),
		T.ToTensor()
	])
	return transforms

class YOLOv3Dataset(data.Dataset):
	def __init__(self, imgDir, numClasses, anchors, S, transforms, args):
		self.annotations = list()
		suffix = (".jpg", ".png", ".jpeg", ".wpeg", ".JPG")
		for f in os.listdir(imgDir):
			annotFile = f.rsplit('.', 1)[0] + ".txt"
			annotFilePath = os.path.join(imgDir, annotFile)
			imgFilePath = os.path.join(imgDir, f)
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

		self.transforms = transformations(args)
	
	def  __len__(self):
		return len(self.annotations)
	
	def __getitem__(self, index):
		bboxes = np.roll(np.loadtxt(fname=self.annotations[index][1], delimiter=" ", ndmin=2), 4, axis=1)
		bboxes = sorted(bboxes, key=lambda x: x[2] * x[3], reverse=True)
		image = Image.open(self.annotations[index][0]).convert("RGB")
		image = self.transforms(image)

		targets = [torch.zeros(self.numAnchorPerScale, S, S, 6) for i, S in enumerate(self.S)]
		for box in bboxes:
			iouAnchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
			anchorIndices = iouAnchors.argsort(descending=True, dim=0)
			x, y, width, height, classLabel = box
			hasAnchor = [False]  * self.numAnchorPerScale
			for anchorIdx in anchorIndices:
				scaleIdx = anchorIdx // self.numAnchorPerScale
				anchorOnScale = anchorIdx % self.numAnchorPerScale
				S = self.S[scaleIdx]
				gridX, gridY = S * x, S * y
				i, j = int(gridY), int(gridX)
				try:
					anchorTaken = targets[scaleIdx][anchorOnScale, i, j, 0]
				except Exception as e:
					print(e, scaleIdx.item(), anchorOnScale.item(), i, j, 0, box, x, y, self.annotations[index][0])
					exit()

				if not anchorTaken and not hasAnchor[scaleIdx]:
					targets[scaleIdx][anchorOnScale, i, j, 0] = 1
					xCell, yCell = gridX - j, gridY - i 

					widthCell, heightCell = S * width, S * height
					targets[scaleIdx][anchorOnScale, i, j, 1:] = torch.tensor(
						[xCell, yCell, widthCell, heightCell, int(classLabel)]
					)
					hasAnchor[scaleIdx] = True
				elif not anchorTaken and iouAnchors[anchorIdx] > self.ignoreIOUThresh:
					targets[scaleIdx][anchorOnScale, i, j, 0] = -1  # ignore prediction
		return image, tuple(targets)


def iou_width_height(boxes1: torch.tensor, boxes2:torch.tensor) -> torch.tensor:
	intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
		boxes1[..., 1], boxes2[..., 1]
	)
	union = (
		boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
	)
	return intersection / union


def intersection_over_union(boxes_preds, boxes_labels):

	box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
	box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
	box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
	box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
	box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
	box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
	box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
	box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

	x1 = torch.max(box1_x1, box2_x1)
	y1 = torch.max(box1_y1, box2_y1)
	x2 = torch.min(box1_x2, box2_x2)
	y2 = torch.min(box1_y2, box2_y2)

	intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
	box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
	box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

	return intersection / (box1_area + box2_area - intersection + 1e-6)


class YOLODetectionLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()
		self.bce = nn.BCEWithLogitsLoss()
		self.entropy = nn.CrossEntropyLoss()
		self.sigmoid = nn.Sigmoid()
	

	def forward(self, predictions, target, anchors):
		obj = target[..., 0] == 1
		noobj = target[..., 0] == 0
		no_object_loss = self.bce(predictions[..., 0][noobj], target[..., 0][noobj])
		anchors = anchors.reshape(1, 3, 1, 1, 2)

		box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
		ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
		object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

		predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
		target[..., 3:5] = torch.log((1e-16 + target[..., 3:5])) / anchors
		box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

		class_loss = self.entropy((predictions[..., 5:][obj]), (target[..., 5][obj].long()))

		return (no_object_loss + object_loss + box_loss + class_loss)


def passInputs(args, model, inputs, criterion, device):
	# print(inputs[0].shape)
	logits = model.forward(inputs[0].to(device))
	loss = (
		criterion(logits[0], inputs[1][0].to(device), args['scaledAnchor'][0]) + 
		criterion(logits[1], inputs[1][1].to(device), args['scaledAnchor'][1]) +
		criterion(logits[2], inputs[1][2].to(device), args['scaledAnchor'][2])
	)
	return loss

def train(args):
	args['anchors'] = args['anchors'][::-1]
	args['S'] = [args['imageSize']//32, args['imageSize']//16, args['imageSize']//8]
	args['scaledAnchor'] = (np.array(args['anchors']) / args['imageSize']).tolist()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# model = Model(anchors=args['scaledAnchor'], numClasses=args['numClasses']).to(device)
	model = ResNetYOLO(anchors=args['scaledAnchor'], numClasses=args['numClasses']).to(device)
	# model.load_state_dict(torch.load( "model.pth"))
	scaler = torch.cuda.amp.GradScaler()
	
	# summary(model, (3, args['imageSize'], args['imageSize']))
	optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.0001)
	# optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=1e-3)

	trainDataset = YOLOv3Dataset(args['trainFolder'], numClasses=args['numClasses'], anchors=args['scaledAnchor'][::-1], S=args['S'], transforms=None, args=args)
	trainLoader = data.DataLoader(trainDataset, batch_size=args['batchSize'], pin_memory=True, shuffle=True)
	# print(len(trainDataset))
	args['scaledAnchor'] = torch.tensor(args['scaledAnchor']).to(device)
	criterion = YOLODetectionLoss()
	trainLossEpochs = []

	for epoch in range(1, args['numEpochs'] + 1):
		model.train()
		trainLossIteration = []
		with tqdm(trainLoader, unit=" batch", desc="Training", leave=False) as tepoch:
			for i, inputs in enumerate(tepoch):
				optimizer.zero_grad()
				with torch.cuda.amp.autocast():
					loss = passInputs(args, model, inputs, criterion, device)
				# loss.backward()
				# optimizer.step()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()

				trainLossIteration.append(loss.item())
				tepoch.set_postfix(loss=trainLossIteration[-1])
			iterationLoss = sum(trainLossIteration) / len(trainLossIteration)
			print(f"Epochs: {epoch}, Loss: {iterationLoss}, lr: {optimizer.param_groups[0]['lr']}")
			torch.save(model.state_dict(), "model.pth")
			
		optimizer.param_groups[0]["lr"] = round(args['lr'] * ((args['numEpochs'] - epoch) / args['numEpochs']), 10)

	model.to("cpu")
	model.load_state_dict(torch.load("model.pth"))
	torch.onnx.export(model, torch.randn(1, 3, 416, 416),  "model.onnx", verbose=True, output_names=["output1", "output2", "output3"])


def save_loss_image(train_loss, val_loss, epoch, model_save_name, model_save_directory):

	fig = plt.figure(figsize=(20, 20))

	plt.scatter([k for k in range(1, epoch + 1)], train_loss, label = "Training Loss")
	# plt.plot([k for k in range(1, epoch + 1)], val_loss, label = "Validation Loss", marker="o")
	plt.legend()
	plt.title(model_save_name)
	fig.canvas.draw()
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	cv2.imwrite(os.path.join(model_save_directory, f"{model_save_name}_loss.jpg"), img)

	plt.close()

if __name__ == "__main__":
	os.environ['TORCH_HOME'] = os.path.sep.join(["E:", "Models", "cache", "pytorch"])
	torch.cuda.empty_cache()
	torch.backends.cudnn.benchmark = True
	args = {
		"anchors": [[[10,13],  [16,30],  [33,23]],  [[30,61],  [62, 45],  [59, 119]],  [[116,90],  [156,198],  [373,326]]],
		# "trainFolder": r"E:\dataset\SOTA\VOCdevkit\VOC2012\JPEGImages",
		"trainFolder": r"E:\dataset\detection\TextDetection\obj",
		"imageSize": 416,
		"numClasses": 1,
		# 'lr': 0.00087,
		'lr': 0.0087,
		'numEpochs': 100,
		'batchSize': 64
	}
	train(args)