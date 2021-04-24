import cv2
import numpy as np


class FaceDetector():
    def __init__(self, model_architecture, model_weights):
        self.net = cv2.dnn.readNetFromCaffe(model_architecture, model_weights)

    def detect_faces(self, image):
        blob = self.prepare_blob(image)
        self.net.setInput(blob)
        return self.net.forward()

    def prepare_blob(self, image):
        image = cv2.resize(image, (300, 300))
        return cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), 
            swapRB=False, crop=False)

    def get_best_detection(self, face_detections):
        return np.argmax(face_detections[0, 0, :, 2])

    def get_face_coordinates(self, image, face_detections, index):
        (height, width) = image.shape[:2]
        face = face_detections[0, 0, index, 3:7] * np.array([width, height, width, height])
        return face.astype('int')

    def crop_face(self, image, face_coordinates):
        (startX, startY, endX, endY) = face_coordinates
        return image[startY:endY, startX:endX]
