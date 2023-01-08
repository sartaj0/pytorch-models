import numpy as np 
import cv2, os , math, sys
import random


np.set_printoptions(threshold=sys.maxsize)
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
net = cv2.dnn.readNetFromONNX(r"saved_models/model2.onnx")
net = cv2.dnn.readNetFromONNX(r"E:\Projects\pytorch-models\saved_models\model.onnx")
# net = cv2.dnn.readNetFromONNX(r"model.onnx")
# net = cv2.dnn.readNetFromONNX(r"E:\Desktop\model.onnx")
# image = cv2.imread(r"E:\dataset\detection\data\trainval\finalDataset\image_000000041.jpg")
# image = cv2.imread(r"E:\dataset\SOTA\VOCdevkit\VOC2012\JPEGImages\2012_003960.jpg")
# folder = r"E:\dataset\detection\data\trainval\finalDataset"
# folder = r"E:\dataset\SOTA\VOCdevkit\VOC2012\JPEGImages"
folder = r"E:\dataset\detection\TextDetection\obj"
# classes = open(r"E:\dataset\SOTA\VOCdevkit\VOC2012\JPEGImages\classes.txt", "r").read().split("\n")
classes = ['table', 'row', 'column','rh', 'th', 'm']
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
imagePath = os.path.join(folder, random.choice(os.listdir(folder)).split(".")[0] + ".jpg")
image = cv2.imread(imagePath)
image = cv2.imread(r"E:\dataset\detection\pubtables\val\images\PMC512288_table_4.jpg")
def process(image, net):
    H, W = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, size=(416, 416), scalefactor=1/255.0, swapRB=True)
    # blob = cv2.dnn.blobFromImage(image, size=(416, 416), scalefactor=1/127.5, mean=(127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(["output1", "output2", "output3"])

    boxes = []
    classIDs = []
    confidences = []
    for i in range(len(outputs)):
        anchors = np.array([
        [[10,13],  [16,30],  [33,23]],  
        [[30,61],  [62, 45],  [59, 119]],  
        [[116,90],  [156,198],  [373,326]]
        ][::-1][i]).reshape(1, 3, 1, 1, 2) / 416.0

        output = outputs[i]
        gridSize = output.shape[2]

        grid = np.arange(gridSize)
        a, b = np.meshgrid(grid, grid)
        grid = np.concatenate((a.reshape(-1, 1), b.reshape(-1, 1)), axis=-1).reshape(gridSize, gridSize, 2)

        output[..., 0] = sigmoid(output[..., 0])
        output[..., 1: 3] = (sigmoid(output[..., 1: 3]) + grid.reshape(1, gridSize, gridSize, 2)) 
        output[..., 3:5] = (np.exp(output[..., 3:5]) * anchors) 

        output[..., 1: 5] = output[..., 1: 5] / gridSize
        bboxes = output[..., ][output[..., 0] > 0.1]
        
        for box in bboxes:
            x, y, w, h = box[1:5] * np.array([W, H, W, H])
            # print(x, y, w, h)
            conf = box[0]
            try:
                x1 = max(0, int(x - (w/2)))
                y1 = max(0,int(y - (h/2)))
                x2 = min(W, int(x + (w/2)))
                y2 = min(H, int(y + (h/2)))
            except Exception as e:
                continue
            boxes.append([x1, y1, int(w), int(h)])
            confidences.append(conf)
            classID = np.argmax(box[5:])
            classIDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            print(x, y, w, h)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
# '''
image = process(image, net)
cv2.imshow("image", image)
cv2.waitKey(0)
'''
# exit()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = process(frame, net)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()

'''