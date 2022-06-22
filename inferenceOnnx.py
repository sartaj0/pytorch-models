import numpy as np 
import cv2, os , math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
net = cv2.dnn.readNetFromONNX("model.onnx")
image = cv2.imread(r"E:\dataset\detection\data\trainval\finalDataset\image_000000095.jpg")
# image = cv2.imread(r"E:\dataset\SOTA\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg")
H, W = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, size=(416, 416), scalefactor=1/255.0, swapRB=True)
net.setInput(blob)
outputs = net.forward(["output1", "output2", "output3"])
for i in range(len(outputs)):
    anchors = np.array([
    [[10,13],  [16,30],  [33,23]],  
    [[30,61],  [62, 45],  [59, 119]],  
    [[116,90],  [156,198],  [373,326]]
    ][::-1][i]).reshape(1, 3, 1, 1, 2) / 416.0
    # print(anchors)

    output = outputs[i]
    # print(output.shape)
    # print(net.getLayerNames())
    gridSize = output.shape[2]

    grid = np.arange(gridSize)
    a, b = np.meshgrid(grid, grid)
    grid = np.concatenate((a.reshape(-1, 1), b.reshape(-1, 1)), axis=-1).reshape(gridSize, gridSize, 2)

    output[..., 0] = sigmoid(output[..., 0])
    output[..., 1: 3] = (sigmoid(output[..., 1: 3]) + grid.reshape(1, gridSize, gridSize, 2)) 
    output[..., 3:5] = (np.exp(output[..., 3:5]) * anchors) 
    # print(output[0, :, :, :, 3:5].shape)

    output[..., 1: 5] = output[..., 1: 5] / gridSize

    print(output[..., 1: 5].shape)
    # output[..., 1: 5] = output[..., 1: 5] * np.array([W, H, W, H]).reshape(1, 1, 1, 1, 4)
    # output[..., 1: 5] = output[..., 1: 5] * 416
    # print(output[..., 1: 5].astype(np.uint8))
    bboxes = output[..., 1:5][output[..., 0] > 0.5]
    # print(bboxes)
    # print(bboxes.shape)
    # image = cv2.resize(image, (416, 416))
    for box in bboxes:
        x, y, w, h = box * np.array([W, H, W, H])
        x1 = int(x - (w//2))
        y1 = int(y - (h//2))
        x2 = int(x + (w//2)) 
        y2 = int(y + (h//2))
        image = cv2.rectangle(image, (x1, y1), (x2, y2), colors[i], 2)
cv2.imshow("image", image)
cv2.waitKey(0)
exit()

