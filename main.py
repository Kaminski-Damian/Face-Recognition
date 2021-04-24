from face_detection.face_detector import FaceDetector
from face_recognition.face_embedder import FaceEmbedder
from face_recognition.svm_model_trainer import SVMModelTrainer
from face_recognition.face_recognizer import FaceRecognizer
from video_stream.video_stream_handler import VideoStreamHandler
import pickle


caffe_model_architecture = 'face_detection/caffe_model/deploy.prototxt.txt'
caffe_model_weights = 'face_detection/caffe_model/res10_300x300_ssd_iter_140000.caffemodel'

torch_embedding_model = 'face_recognition/torch_model/openface_nn4.small2.v1.t7'

face_detector = FaceDetector(caffe_model_architecture, caffe_model_weights)
face_embedder = FaceEmbedder(torch_embedding_model, face_detector)

if False:
    dataset = 'face_recognition/dataset'
    embedding_min_confidence = 0.8
    embeddings = face_embedder.get_embeddings(dataset, embedding_min_confidence)
    
    svm_model_trainer  = SVMModelTrainer()
    svm_model_trainer.train_svm_model(embeddings)

label_encoder_pickle = 'face_recognition/svm_model/label_encoder.pickle'
svm_model_pickle = 'face_recognition/svm_model/svm_model.pickle'

label_encoder = pickle.loads(open(label_encoder_pickle, 'rb').read())
svm_model = pickle.loads(open(svm_model_pickle, 'rb').read())

face_recognizer = FaceRecognizer(label_encoder, svm_model)
video_stream_handler = VideoStreamHandler(face_detector, face_embedder, face_recognizer)

video_min_confidence = 0.8
video_stream_handler.stream_video(video_min_confidence)
