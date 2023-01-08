import torch
import torch.nn as nn
import torch.nn.functional as F 

import collections


class ConvBatchLeakyReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super(ConvBatchLeakyReLU, self).__init__()
		self.convBatch = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
			nn.BatchNorm2d(num_features=out_channels),
			nn.LeakyReLU(0.1)
		)
	def forward(self, x):
		return self.convBatch(x)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super(ResidualBlock, self).__init__()
	
	def forward(self, x):
		pass

class YOLO(nn.Module):
	def __init__(self):
		super(YOLO, self).__init__()

		self.darknet = nn.Sequential(
			ConvBatchLeakyReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool2d(kernel_size=2, stride=2),


			ConvBatchLeakyReLU(in_channels=64, out_channels=192, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),


			ConvBatchLeakyReLU(in_channels=192, out_channels=128, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=128, out_channels=256, kernel_size=3, padding=1),
			ConvBatchLeakyReLU(in_channels=256, out_channels=256, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),


			ConvBatchLeakyReLU(in_channels=512, out_channels=256, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, padding=1),
			ConvBatchLeakyReLU(in_channels=512, out_channels=256, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, padding=1),
			ConvBatchLeakyReLU(in_channels=512, out_channels=256, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, padding=1),
			ConvBatchLeakyReLU(in_channels=512, out_channels=256, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=256, out_channels=512, kernel_size=3, padding=1),

			ConvBatchLeakyReLU(in_channels=512, out_channels=512, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),


			ConvBatchLeakyReLU(in_channels=1024, out_channels=512, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
			ConvBatchLeakyReLU(in_channels=1024, out_channels=512, kernel_size=1),
			ConvBatchLeakyReLU(in_channels=512, out_channels=1024, kernel_size=3, padding=1),

			ConvBatchLeakyReLU(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
			ConvBatchLeakyReLU(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),


			ConvBatchLeakyReLU(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
			ConvBatchLeakyReLU(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
		)

		self.last_layer = nn.Sequential(
			nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.5),
			nn.Conv2d(in_channels=1, out_channels=30, kernel_size=1)
		)

		# self.last_layer = nn.Sequential(
		# 	nn.Flatten(),
		# 	nn.Linear(in_features=1024 * 7 * 7, out_features=496),
		# 	nn.LeakyReLU(0.1),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(in_features=496, out_features=7 * 7 * (5 * 2 + 20))
		# )
	
	def forward(self, x):
		x = self.darknet(x)
		x = self.last_layer(x)
		x = x.view(-1, 30, 7, 7)
		x = x.permute(0, 2, 3, 1)
		return x

if __name__ == "__main__":
	from torchsummary import summary

	model = YOLO()
	model.to("cuda")
	# summary(model, (3, 448, 448))
	
	a = torch.randn(1, 3, 448, 448).to("cuda")
	print(model(a).shape)
	# print(model)
	torch.save(model.state_dict(), "model.pth")