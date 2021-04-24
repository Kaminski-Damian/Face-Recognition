from imutils.video import VideoStream
import time
import imutils
import cv2


class VideoStreamHandler:
    video_stream = None

    def __init__(self, face_detector, face_embedder, face_recognizer):
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.face_recognizer = face_recognizer

    def stream_video(self, min_confidence):
        self.turn_on_camera()

        while True:
            frame = self.prepatre_frame()
            face_detections = self.face_detector.detect_faces(frame)

            for index in range(0, face_detections.shape[2]):
                confidence = face_detections[0, 0, index, 2]
                if confidence < min_confidence:
                    continue

                face_coordinates = self.face_detector.get_face_coordinates(frame, face_detections, index)
                cropped_face = self.face_detector.crop_face(frame, face_coordinates)
                
                (cropped_face_height, cropped_face_width) = cropped_face.shape[:2]
                if cropped_face_height < 40 or cropped_face_width < 40:
                    continue                

                blob = self.face_embedder.prepare_blob(cropped_face)
                vector = self.face_embedder.get_vector_from_blob(blob)
                recognized_face = self.face_recognizer.recognize_face(vector)

                self.draw_face_box(frame, face_coordinates, recognized_face)

            cv2.imshow('Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.turn_off_camera()
                
    def turn_on_camera(self):
        print('[+] Start video streaming')
        self.video_stream = VideoStream(src=0).start()
        time.sleep(1.0)

    def prepatre_frame(self):
        frame = self.video_stream.read()
        return imutils.resize(frame, width=600)

    def draw_face_box(self, frame, face_coordinates, recognized_face):
        (startX, startY, endX, endY) = face_coordinates
        recognition_probability = recognized_face['proba'] * 100
        recognition_label = recognized_face['label']

        text = '{}: {:.2f}%'.format(recognition_label, recognition_probability)  
        textY = startY - 10 if startY - 10 > 10 else startY + 10
        bgr = (0, 255, 0) if recognition_label == 'damian' else (255, 255, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), bgr, 2)
        cv2.putText(frame, text, (startX, textY), cv2.FONT_HERSHEY_SIMPLEX , 0.45, bgr, 2)

    def turn_off_camera(self):
        cv2.destroyAllWindows()
        self.video_stream.stop()
        print('[+] Video streaming stopped')
