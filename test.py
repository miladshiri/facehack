import cv2
from facehack.api import Faces

# frame = cv2.imread('test7small.png')
frame = cv2.imread('/home/khorshidsoft/workspace/data/a2.jpg')
faces_obj = Faces(frame)
all_faces = faces_obj.detect_all()

print (len(all_faces))
for face in all_faces:
    face.show_landmarks()
