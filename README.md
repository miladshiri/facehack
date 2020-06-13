# FaceHack

A python library to easily detect and work with human faces. 

## Detecting All Faces in an Image

```
import cv2
from facehack.api import Faces

image = cv2.imread('test.jpg')
f_obj = Faces(frame)
all_faces = f_obj.detect_all()

```

## Working with each face separately

```
for face in all_faces:
    ## Visulaize landmarks on the face image
    face.show_landmarks()
    
    ## Face shpe estimation
    print(face.detect_shape())
    
    ## Face skin color estimation
    print (face.detect_skin_color())
    
    ## Eye color estimation
    print (face.detect_eye_color())

```
