import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from models.yolov3 import YOLOv3
from modules.loss import YOLODetectionLoss
from modules.dataset import YOLOv3Dataset
from modules.augmentations import trainTranforms, testTranforms
from modules.utils import mean_average_precision, cells_to_bboxes, get_evaluation_bboxes, check_class_accuracy, convert
from models.resenetYolo import ResNetYOLO

import os
import time
import numpy as np
from tqdm import tqdm
from modules.utils import save_loss_image
from modules.parse_opt import parse_opt

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
	args["anchors"] = [[[10,13],  [16,30],  [33,23]],  [[30,61],  [62, 45],  [59, 119]],  [[116,90],  [156,198],  [373,326]]][::-1]
	# args["anchors"] = [[[8,7],  [21,7],  [18,13]],  [[51,13],  [202,13],  [60,87]],  [[356,15],  [119,178],  [105,323]]][::-1],
	torch.backends.cudnn.benchmark = True

	if not os.path.isdir(args['checkpointFolder']):
		os.mkdir(args['checkpointFolder'])
	pthFileName  = os.path.join(args['checkpointFolder'], args['checkpointFile']+".pth")
	onnxFileName = os.path.join(args['checkpointFolder'], args['checkpointFile']+".onnx")
	S = [args['imageSize']//32, args['imageSize']//16, args['imageSize']//8]
	args['scaledAnchor'] = (np.array(args['anchors']) / args['imageSize']).tolist()

	if args['valFolder'] is not None:
		trainDataset = YOLOv3Dataset(args['trainFolder'], numClasses=args['numClasses'], anchors=args['scaledAnchor'], S=S, transforms=trainTranforms, args=args)
		trainLoader = data.DataLoader(trainDataset, batch_size=args['batchSize'], pin_memory=True, 
			shuffle=False, num_workers=4)

		valDataset = YOLOv3Dataset(args['valFolder'], numClasses=args['numClasses'], anchors=args['scaledAnchor'], S=S, transforms=testTranforms, args=args)
		valLoader = data.DataLoader(valDataset, batch_size=args['batchSize'], pin_memory=True, 
			shuffle=False, num_workers=4)

	if args['valFolder'] is None:

		trainDataset = YOLOv3Dataset(args['trainFolder'], numClasses=args['numClasses'], anchors=args['scaledAnchor'], S=S, transforms=trainTranforms, args=args)
		valid_size = 0.2
		num_train = len(trainDataset)
		indices = list(range(num_train))
		split = int(np.floor(valid_size * num_train))
		np.random.shuffle(indices)
		train_idx, test_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		test_sampler = SubsetRandomSampler(test_idx)
		trainLoader = torch.utils.data.DataLoader(trainDataset, sampler=train_sampler, 
			batch_size=args['batchSize'], pin_memory=True, num_workers=4)
		valLoader = torch.utils.data.DataLoader(trainDataset, sampler=test_sampler, 
			batch_size=args['batchSize'], pin_memory=True, num_workers=4)


	model = ResNetYOLO(anchors=args['anchors'], numClasses=args['numClasses'])
	# model = YOLOv3(anchors=args['anchors'], numClasses=args['numClasses'])
	# model.load_state_dict(torch.load(pthFileName))
	if args['optimizer'] == "adam":
		optimizer = optim.Adam(model.parameters(), lr=args['lr'])
	elif args['optimizer'] == "sgd":
		optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.937)

	changeAterIteration = round(args['epochs'] * len(trainLoader) / 4)
	base_lr = 0.01
	lrScheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=base_lr/10, 
		step_size_up=changeAterIteration, mode="exp_range")
	numIterations = 0

	# lrScheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainLoader), epochs=args['epochs'])

	# lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args['patience'], mode='min', factor=0.1, min_lr=0.000001)
	# lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
	# lrScheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args['lr'] / 100, max_lr=args['lr'], step_size_up=1,mode="triangular2")

	scaler = torch.cuda.amp.GradScaler()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	criterion = YOLODetectionLoss()

	args['scaledAnchor'] = torch.tensor(args['scaledAnchor']).to(device)
	trainLossEpochs, valLossEpochs = [], []

	for epoch in range(1, args['epochs'] + 1):
		model.train()
		trainLossIteration, valLossIteration = [], []
		

		start = time.time()
		with tqdm(trainLoader, unit=" batch", desc=f"Training {epoch}/{args['epochs']}", leave=False) as tepoch:
			for i, inputs in enumerate(tepoch):
				# optimizer.zero_grad()
				print(inputs[0].min(), inputs[0].max())
				with torch.cuda.amp.autocast():
					loss = passInputs(args, model, inputs, criterion, device)
				# loss.backward()
				# optimizer.step()

				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()

				optimizer.zero_grad()

				trainLossIteration.append(loss.detach().cpu().numpy().round(decimals=5))
				tepoch.set_postfix(loss=trainLossIteration[-1], lr=optimizer.param_groups[0]["lr"])

				numIterations += 1
				if numIterations % changeAterIteration == 0:
					base_lr = base_lr / 10
					lrScheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=base_lr/10, 
					step_size_up=changeAterIteration, mode="exp_range")

				lrScheduler.step()

			trainLossEpochs.append(sum(trainLossIteration) / len(trainLossIteration))

		model.eval()
		with torch.no_grad():
			with tqdm(valLoader, unit=" batch", desc=f"Testing {epoch}/{args['epochs']}", leave=False) as tepoch:
				for i, inputs in enumerate(tepoch):
					with torch.cuda.amp.autocast():
						loss = passInputs(args, model, inputs, criterion, device)

					valLossIteration.append(loss.detach().cpu().numpy().round(decimals=5))
					tepoch.set_postfix(loss=valLossIteration[-1])

				valLossEpochs.append(sum(valLossIteration) / len(valLossIteration))
		timeTaken = time.time() - start
		timeTaken = convert(timeTaken)
		print(f"Epochs: {epoch}, Train Loss: {trainLossEpochs[-1]}, Val Loss: {valLossEpochs[-1]}, Epoch Time: {timeTaken}")
		torch.save(model.state_dict(), pthFileName)
		# print(trainLossEpochs, valLossEpochs)
		save_loss_image(trainLossEpochs, valLossEpochs, epoch, args['checkpointFile'], args['checkpointFolder'])
		
		# optimizer.param_groups[0]["lr"] = round(args['lr'] * ((args['epochs'] - epoch) / args['epochs']), 10)
		# lrScheduler.step(valLossEpochs[-1])
		# lrScheduler.step()
		
	model.to("cpu")
	model.load_state_dict(torch.load(pthFileName))
	model.det = True
	
	torch.onnx.export(model, torch.randn(1, 3, 416, 416),  onnxFileName, 
		input_names=["input"],
		verbose=True, output_names=["output1", "output2", "output3"], opset_version=11)

if __name__ == '__main__':
	os.environ['TORCH_HOME'] = os.path.sep.join(["E:", "Models", "cache", "pytorch"])
	torch.cuda.empty_cache()

	args = parse_opt()
	print(args)
	main(args)

	# print(args)