from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


class SVMModelTrainer:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def train_svm_model(self, embeddings):
        labels = self.label_encoder.fit_transform(embeddings['labels'])
        svm_model = SVC(C=1.0, kernel='linear', probability=True)
        svm_model.fit(embeddings['embeddings'], labels)

        self.save_to_file('label_encoder', self.label_encoder)
        self.save_to_file('svm_model', svm_model)

        print('[+] SVM model trained')

    def save_to_file(self, file_name, data):
        path = 'face_recognition/svm_model/{}.pickle'.format(file_name)
        file = open(path, 'wb')
        file.write(pickle.dumps(data))
        file.close()
