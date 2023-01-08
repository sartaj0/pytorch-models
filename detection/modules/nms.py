import torch
import torch.nn as nn 
import torch.nn.functional as F


def NMS(P: torch.tensor, thresh_iou: float):
	
	x1 = P[:, 0]
	y1 = P[:, 1]
	x2 = P[:, 2]
	y2 = P[:, 3]

	scores = P[:, 4]

	areas = (x2 - x1) * (y2 - y1)
	order = scores.argsort()

	keep = []
	while len(order) > 0:
		idx = order[-1]
		keep.append(P[idx])
		order = order[:-1]
		xx1 = torch.index_select(x1,dim = 0, index = order)
		xx2 = torch.index_select(x2,dim = 0, index = order)
		yy1 = torch.index_select(y1,dim = 0, index = order)
		yy2 = torch.index_select(y2,dim = 0, index = order)
		print(idx, P, order)
		print(xx1, yy1, xx2, yy2)
		xx1 = torch.max(xx1, x1[idx])
		yy1 = torch.max(yy1, y1[idx])
		xx2 = torch.min(xx2, x2[idx])
		yy2 = torch.min(yy2, y2[idx])

		w = xx2 - xx1
		h = yy2 - yy1 

		w = torch.clamp(w, min=0.0)
		h = torch.clamp(h, min=0.0)

		inter = w*h

		rem_areas = torch.index_select(areas, dim = 0, index = order) 

		union = (rem_areas - inter) + areas[idx]

		IoU = inter / union

		mask = IoU < thresh_iou
		order = order[mask]
	return keep

if __name__ == "__main__":
	bboxes = torch.tensor([
		[1, 1, 3, 3, 0.95],
		[1, 1, 3, 4, 0.93],
		[1, 0.9, 3.6, 3, 0.98],
		[1, 0.9, 3.5, 3, 0.97]
	])
	print(NMS(bboxes, 0.5))