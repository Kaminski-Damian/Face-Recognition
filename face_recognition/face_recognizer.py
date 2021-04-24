import numpy as np


class FaceRecognizer:
    def __init__(self, label_encoder, svm_model):
        self.label_encoder = label_encoder
        self.svm_model = svm_model

    def recognize_face(self, vector):
        predicted_values = self.svm_model.predict_proba(vector)[0]
        proba_index = np.argmax(predicted_values)
        proba = predicted_values[proba_index]
        label = self.label_encoder.classes_[proba_index]
        return {
            'proba': proba,
            'label': label      
        }