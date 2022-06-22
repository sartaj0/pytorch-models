import torch
import torch.optim as optim
from torch.utils import data

from models.yolov3 import YOLOv3
from modules.loss import YOLODetectionLoss
from modules.dataset import YOLOv3Dataset
from modules.augmentations import trainTranforms, testTranforms
from modules.utils import mean_average_precision, cells_to_bboxes, get_evaluation_bboxes, check_class_accuracy

import os
from models.resenetYolo import ResNetYOLO
import numpy as np
from tqdm import tqdm

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

def passInputs(args, model, inputs, criterion, device):
	# print(inputs[0].shape)
	logits = model.forward(inputs[0].to(device))
	loss = (
		criterion(logits[0], inputs[1][0].to(device), args['scaledAnchor'][0]) + 
		criterion(logits[1], inputs[1][1].to(device), args['scaledAnchor'][1]) +
		criterion(logits[2], inputs[1][2].to(device), args['scaledAnchor'][2])
	)
	return loss


def main(args):
	S = [args['imageSize']//32, args['imageSize']//16, args['imageSize']//8]
	args['scaledAnchor'] = (np.array(args['anchors']) / args['imageSize']).tolist()
	# args['scaledAnchor'] = args['scaledAnchor']
	
	trainDataset = YOLOv3Dataset(args['trainFolder'], numClasses=2, anchors=args['scaledAnchor'], S=S, transforms=trainTranforms, args=args)
	trainLoader = data.DataLoader(trainDataset, batch_size=args['batchSize'], pin_memory=True, shuffle=True)
	model = ResNetYOLO(anchors=args['anchors'], numClasses=args['numClasses'])
	# model.load_state_dict(torch.load("model.pth"))
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	criterion = YOLODetectionLoss()

	args['scaledAnchor'] = torch.tensor(args['scaledAnchor']).to(device)
	trainLossEpochs = []

	for epoch in range(1, args['numEpochs'] + 1):
		model.train()
		trainLossIteration = []
		with tqdm(trainLoader, unit=" batch", desc="Training", leave=False) as tepoch:
			for i, inputs in enumerate(tepoch):
				optimizer.zero_grad()
				loss = passInputs(args, model, inputs, criterion, device)
				loss.backward()
				optimizer.step()

				trainLossIteration.append(loss.item())
				tepoch.set_postfix(loss=trainLossIteration[-1])
			iterationLoss = sum(trainLossIteration) / len(trainLossIteration)
			print(f"Epochs: {epoch}, Loss: {iterationLoss}, lr: {optimizer.param_groups[0]['lr']}")
			torch.save(model.state_dict(), "model.pth")
		
		optimizer.param_groups[0]["lr"] = round(args['lr'] - args['lr']*(1/(args['numEpochs']))*epoch, 10)
	model.to("cpu")
	model.load_state_dict(torch.load("model.pth"))
	torch.onnx.export(model, torch.randn(1, 3, 416, 416),  "model.onnx", verbose=True, output_names=["output1", "output2", "output3"])

if __name__ == '__main__':
	os.environ['TORCH_HOME'] = os.path.sep.join(["E:", "Models", "cache", "pytorch"])
	args = {
		"lr": 0.0001,
		"weightDecay": 1e-3,
		"trainFolder": r"E:\dataset\SOTA\VOCdevkit\VOC2012\JPEGImages",
		"imageSize": 416,
		"numEpochs": 15,
		"confThresh": 0.6,
		"mapIOUThresh": 0.5,
		"nmsIOUThresh": 0.45,
		"checkpointFolder": "",
		"checkpointFile": ".pth",
		"batchSize": 32,
		"anchors": [[[10,13],  [16,30],  [33,23]],  [[30,61],  [62, 45],  [59, 119]],  [[116,90],  [156,198],  [373,326]]][::-1],
		"numClasses": 20
	}
	main(args)

	# print(args)