import cv2
import numpy as np
import dlib 
import os

class Face():
    def __init__(self, face, mode='dlib'):
        self.face = face
        self.mode = mode

    def show(self):
        cv2.imshow('face', self.face)
        cv2.waitKey(0)

    @property
    def landmarks(self):
        if self.mode == 'dlib':
            full_path = os.path.realpath(__file__)
            path, filename = os.path.split(full_path)
            predictor = dlib.shape_predictor(path + "/resource/shape_predictor_68_face_landmarks.dat")
            box = dlib.rectangle(0, 0, self.face.shape[1], self.face.shape[0])
            result = predictor(self.face, box)
            landmarks = []
            for n in range(0, 68):
                x = result.part(n).x
                y = result.part(n).y
                landmarks.append((x, y))
                # cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        elif mode == 'opencv':
            pass
        else:
            raise Exception ('landmark detector mode is unknown')
        
        return landmarks

    def show_landmarks(self):
        face_copy = self.face.copy()
        for landmark in self.landmarks:
            cv2.circle(face_copy, (landmark[0], landmark[1]), 2, (255, 0, 0), -1)
        cv2.imshow('landmarks', face_copy)
        cv2.waitKey(0)


class Faces():
    def __init__(self, image, mode='dlib'):
        self.image = image
        self.mode = mode

    def detect_all(self):
        if self.mode == 'dlib':
            detector = dlib.get_frontal_face_detector()
            faces = detector(self.image)
            padding = 0
            result = []
            for face in faces:
                face_cropped = self.image[face.top()-padding:face.bottom()+padding, face.left()-padding:face.right()+padding]
                result.append(Face(face_cropped))

        else:
            raise Exception ('face detector mode is unknown')

        return result
