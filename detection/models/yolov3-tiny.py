from layers import *

import torch
import torch.nn as nn 


class TinyYOLOv3(nn.Module):
	def __init__(self):
		super(TinyYOLOv3, self).__init__()
		self.anchors = [[[10,14],  [23,27],  [37,58]],  [[81,82],  [135,169],  [344,319]]]
		self.conv1 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(kernel_size=2),

			ConvBatchLeakyReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(kernel_size=2),

			ConvBatchLeakyReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(kernel_size=2),

			ConvBatchLeakyReLU(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(kernel_size=2),

			ConvBatchLeakyReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
		)

		self.conv2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2),

			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			# nn.MaxPool2d(kernel_size=2, stride=1),

			ConvBatchLeakyReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
			ConvBatchLeakyReLU(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1),
		)

		self.output1 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, stride=1, padding=0),
			DetectionLayer(self.anchors[1])
		)
		self.conv3 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
			nn.UpsamplingBilinear2d(scale_factor=2)
		)
		self.output2 = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1, stride=1, padding=0),
			DetectionLayer(self.anchors[0])
		)
	def forward(self, x):
		x1 = self.conv1(x)
		x = self.conv2(x1)
		output1 = self.output1(x)
		x = self.conv3(x)
		print(x.shape, x1.shape)
		x = torch.cat((x1, x), dim=1)
		output2 = self.output2(x)

		return output1, output2


if __name__ == "__main__":
	from torchsummary import summary

	model = TinyYOLOv3()
	# print(model)
	model.to("cuda")
	# summary(model, (3, 416, 416))
	
	a = torch.randn(1, 3, 416, 416).to("cuda")
	print([i.shape for i in model(a)])
	# print(model)
	torch.save(model.state_dict(), "model.pth")
	# model.to("cpu")
	# a = a.to("cpu")
	# torch.onnx.export(model, a,  "model.onnx", verbose=True, output_names=["output1"])

