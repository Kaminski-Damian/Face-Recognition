import cv2
import imutils 
from imutils import paths
import os


class FaceEmbedder:
    def __init__(self, embedding_model, face_detector):
        self.net = cv2.dnn.readNetFromTorch(embedding_model)
        self.face_detector = face_detector

    def get_embeddings(self, dataset, min_confidence):
        image_paths = list(paths.list_images(dataset))

        embeddings = []
        labels = []

        for (i, image_path) in enumerate(image_paths):
            print('[+] Processing {}/{}'.format(i + 1, len(image_paths)))

            image = self.prepare_image_from_image_path(image_path)
            face_detections = self.face_detector.detect_faces(image)
            
            if len(face_detections) > 0:
                index = self.face_detector.get_best_detection(face_detections)
                confidence = face_detections[0, 0, index, 2]

                if confidence >= min_confidence:
                    face_coordinates = self.face_detector.get_face_coordinates(image, face_detections, index) 
                    cropped_face = self.face_detector.crop_face(image, face_coordinates)
                    blob = self.prepare_blob(cropped_face)

                    vector = self.get_vector_from_blob(blob)
                    label = self.get_label_from_image_path(image_path)

                    embeddings.append(vector.flatten())
                    labels.append(label)

        return {
            'embeddings': embeddings,
            'labels': labels
        }

    def prepare_image_from_image_path(self, image_path):
        image = cv2.imread(image_path)
        return imutils.resize(image, width=600)

    def prepare_blob(self, cropped_face):
        return cv2.dnn.blobFromImage(cropped_face, 1.0 / 255, (96, 96), (0, 0, 0),
            swapRB=True, crop=False)

    def get_vector_from_blob(self, blob):
        self.net.setInput(blob)
        return self.net.forward()

    def get_label_from_image_path(self, image_path):
        return image_path.split(os.path.sep)[-2]
