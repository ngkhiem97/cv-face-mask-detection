import os
import numpy as np
import cv2

def main():
    # Load the cascade
    directory = os.path.dirname(__file__)
    #capture = cv2.VideoCapture(os.path.join(directory, "image.jpg")) # For a single image
    capture = cv2.VideoCapture(0) # For webcam
    if not capture.isOpened():
        exit()
    
    # Load the face detector model
    weights = os.path.join(directory, "yunet.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

    while True:
        # Load single frame
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        # 検出した顔のバウンディングボックスとランドマークを描画する
        for face in faces:
            # バウンディングボックス
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # ランドマーク（右目、左目、鼻、右口角、左口角）
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)
                
            # 信頼度
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

        # 画像を表示する
        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()