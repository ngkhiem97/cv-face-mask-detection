import os
import numpy as np
import cv2
import torch
from models.alexnet import AlexNet
import json
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def main():
    directory = os.path.dirname(__file__)
    #capture = cv2.VideoCapture(os.path.join(directory, "image.jpg")) # For a single image
    capture = cv2.VideoCapture(0) # For webcam
    if not capture.isOpened():
        exit()
    
    # YuNet
    weights = os.path.join(directory, "./models/yunet.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

    # AlexNet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    json_path = './categories.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    model = AlexNet(num_classes=2).to(device)
    weights_path = "./models/AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            box = list(map(int, face[:4]))

            xmin = box[0]
            ymin = box[1]
            xmax = box[0] + box[2]
            ymax = box[1] + box[3]
            image_data = Image.fromarray(image[:,:,::-1])
            image_crop = image_data.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
            image_crop.save("./images/face_crop.jpg")
            image_crop = data_transform(image_crop)
            image_crop = torch.unsqueeze(image_crop, dim=0)
            with torch.no_grad():
                output = torch.squeeze(model(image_crop.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            print_res = "i:{}   class: {}   prob: {:.3}".format(predict_cla, 
                                                                class_indict[str(predict_cla)],
                                                                predict[predict_cla].numpy())
            position_2 = (box[0], box[1] + box[3] + 20)
            color = (0, 255, 0) if predict_cla == 1 else (0, 0, 255)
            scale = 0.5
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, print_res, position_2, font, scale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)
                
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            thickness = 2
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()