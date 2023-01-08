import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torchvision import models

from .layers import *

class ResNetYOLO(nn.Module):
	def __init__(self, anchors, numClasses):
		super(ResNetYOLO, self).__init__()

		self.numClasses = numClasses
		self.anchorPerScale = 3
		self.filter_size = self.anchorPerScale * (self.numClasses + 5)
		self.anchors = torch.tensor(anchors) / 416
		self.conv1 = ConvBatchLeakyReLU(in_channels=3, out_channels=32, kernel_size=3, padding=1)
		self.upsampleMode = "nearest"
		self.det = False

		requires_grad_ = False
		model =  models.resnet34(pretrained=True)
		self.conv1 = nn.Sequential(
			*list(model.children())[:5],
		).requires_grad_(requires_grad_)
		self.conv2 = model.layer2.requires_grad_(requires_grad_)
		self.conv3 = model.layer3.requires_grad_(requires_grad_)
		self.conv4 = model.layer4.requires_grad_(requires_grad_)

		self.dcc1 = nn.Sequential(
			DoubleConvBatch(512, 256, 512),
			DoubleConvBatch(512, 256, 512),
			ConvBatchLeakyReLU(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
		)
		self.output1 = OutputLayer(in_channels=512, mid_channels=512, anchors=self.anchors[0], 
		numClasses=self.numClasses, anchorPerScale=self.anchorPerScale)

		self.convUnPool1 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
			nn.Upsample(scale_factor=2, mode=self.upsampleMode)
		)

		self.dcc2 = nn.Sequential(
			DoubleConvBatch(512, 128, 256),
			DoubleConvBatch(256, 128, 256),
			ConvBatchLeakyReLU(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
		)
		self.output2 = OutputLayer(in_channels=256, mid_channels=256, anchors=self.anchors[1], 
		numClasses=self.numClasses, anchorPerScale=self.anchorPerScale)

		self.convUnPool2 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1),
			nn.Upsample(scale_factor=2, mode=self.upsampleMode)
		)

		self.dcc3 = nn.Sequential(
			DoubleConvBatch(256, 64, 128),
			DoubleConvBatch(128, 64, 128),
			ConvBatchLeakyReLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
		)

		self.output3 = OutputLayer(in_channels=128, mid_channels=128, anchors=self.anchors[2], 
		numClasses=self.numClasses, anchorPerScale=self.anchorPerScale)

		self.det1 = DetectionLayer(13, self.anchors[0])
		self.det2 = DetectionLayer(26, self.anchors[1])
		self.det3 = DetectionLayer(52, self.anchors[2])

	def forward(self, x):
		x = self.conv1(x) #64x104x104
		x1 = self.conv2(x) #128x52x52
		x2 = self.conv3(x1) #256x26x26
		x = self.conv4(x2) #512x13x13

		x = self.dcc1(x) #512x13x13
		output1 = self.output1(x) #13x13x-1
		x = self.convUnPool1(x) #256x26x26
		x = torch.cat((x2, x), 1)
		x = self.dcc2(x)
		output2 = self.output2(x) #26x26x-1
		x = self.convUnPool2(x) #128x52x52
		x = torch.cat((x1, x), 1) #256x52x52
		x = self.dcc3(x) #128x52x52
		output3 = self.output3(x)

		# output1 = self.det1(output1)
		# output2 = self.det2(output2)
		# output3 = self.det3(output3)

		return (output1, output2, output3)



if __name__ == "__main__":
	os.environ['TORCH_HOME'] = os.path.sep.join(["E:", "Models", "cache", "pytorch"])
	anchor = torch.tensor([[[10,13],  [16,30],  [33,23]],  [[30,61],  [62, 45],  [59, 119]],  [[116,90],  [156,198],  [373,326]]][::-1]) / 416.0

	net = ResNetYOLO(anchors=anchor, numClasses=2)
	print(net)
	print([x.shape for x in net(torch.randn(1, 3, 416, 416))])
