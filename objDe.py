import torch
from torchvision import models, transforms
from PIL import ImageTk, Image
import cv2

'''
# Load the pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Save the model to disk
torch.save(model.state_dict(), "fasterrcnn_resnet50_fpn.pth")
'''
# Load the model from disk
model = models.detection.fasterrcnn_resnet50_fpn()
model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn.pth"))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Define the COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def detect_objects(image):

    # convert image to a tensor
    image = torch.from_numpy(image)

    # unsqueeze to add a batch dimension
    image = torch.unsqueeze(image, 0)

    # Apply the model to the image
    with torch.no_grad():
        output = model([image])
    # Extract the bounding boxes and labels from the output
    boxes = output[0]['boxes'].detach().numpy()
    labels = output[0]['labels'].detach().numpy()
    scores = output[0]['scores'].detach().numpy()
    # Draw the bounding boxes and labels on the image
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            box = box.astype(int)
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(image, label_name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX)
    return image


'''

# Read the image and apply the transform
image = Image.open("car.jpeg")
image = transform(image).unsqueeze(0)

# Perform the object detection
output = model(image)

# Extract the bounding boxes and labels from the output
boxes = output[0]['boxes'].detach().numpy()
labels = output[0]['labels'].detach().numpy()
scores = output[0]['scores'].detach().numpy()

# Print the object name and highest prediction for each object
for i in range(3):
    label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
    score = scores[i]
    print("Object: ", label)
    print("Confidence score: ", score)
    print("\n")

# Print the object name and highest prediction for each object
for i in range(len(boxes)):
    label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
'''