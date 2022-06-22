import torch
import torch.nn as nn

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


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels):
		super(ResidualBlock, self).__init__()

		self.doubleConv = DoubleConvBatch(in_channels, mid_channels, out_channels)
	
	def forward(self, x):
		x = torch.add(self.doubleConv(x), x)
		return x