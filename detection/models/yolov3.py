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

class YOLOv3(nn.Module):
	def __init__(self, anchors, numClasses):
		super(YOLOv3, self).__init__()

		self.numClasses = numClasses
		self.anchorPerScale = 3
		self.filter_size = self.anchorPerScale * (self.numClasses + 5)
		self.anchors = anchors
		self.conv1 = ConvBatchLeakyReLU(in_channels=3, out_channels=32, kernel_size=3, padding=1)
		self.upsampleMode = "nearest"

		self.convres1 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
			ResidualBlock(in_channels=64, mid_channels=32, out_channels=64)
		)

		self.convres2 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=128, mid_channels=64, out_channels=128) for i in range(2)]
		)

		self.convres3 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=256, mid_channels=128, out_channels=256) for i in range(8)],
		)

		self.convres4 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
			*[ResidualBlock(in_channels=512, mid_channels=256, out_channels=512) for i in range(8)],
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


if __name__ == "__main__":
	from torchsummary import summary

	model = YOLOv3()
	print(model)
	model.to("cuda")
	# summary(model, (3, 416, 416))
	
	a = torch.randn(1, 3, 416, 416).to("cuda")
	print([i.shape for i in model(a)])
	# print(model)
	torch.save(model.state_dict(), "model.pth")
	# model.to("cpu")
	# a = a.to("cpu")
	# torch.onnx.export(model, a,  "model.onnx", verbose=True, output_names=["output1"])