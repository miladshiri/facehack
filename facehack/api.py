import cv2
import numpy as np
import dlib 
import os
from sklearn.cluster import KMeans

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
            landmarks = predictor(self.face, box)
            # landmarks = []
            # for n in range(0, 68):
            #     x = result.part(n).x
            #     y = result.part(n).y
            #     landmarks.append((x, y))
                # cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        elif mode == 'opencv':
            pass
        else:
            raise Exception ('landmark detector mode is unknown')
        
        return landmarks

    def show_landmarks(self):
        face_copy = self.face.copy()
        for landmark in self.landmarks.parts():
            cv2.circle(face_copy, (landmark.x, landmark.y), 2, (255, 0, 0), -1)
        cv2.imshow('landmarks', face_copy)
        cv2.waitKey(0)


    def detect_shape(self):
        image = self.face
        detected_landmarks = self.landmarks.parts()
        # x,y,w,h = face
        x = 0
        y = 0
        w = image.shape[1]
        h = image.shape[0]
        original = image.copy()
        # for (x,y,w,h) in faces:
        #draw a rectangle around the faces
        # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        #converting the opencv rectangle coordinates to Dlib rectangle

        #converting to np matrix
        landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
        #making another copy  for showing final results
        results = original.copy()
        # (x,y,w,h) = face
        x = np.int32(x)
        y = np.int32(y)
        w = np.int32(w)
        h = np.int32(h)
        # for (x,y,w,h) in faces:
        #draw a rectangle around the faces
        # cv2.rectangle(results, (x,y), (x+w,y+h), (0,255,0), 2)
        #making temporary copy
        temp = original.copy()
        #getting area of interest from image i.e., forehead (25% of face)
        forehead = temp[y:y+int(0.25*h), x:x+w]
        rows,cols, bands = forehead.shape
        X = forehead.reshape(rows*cols,bands)

        #kmeans
        kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(X)
        for i in range(0,rows):
            for j in range(0,cols):
                if y_kmeans[i*cols+j]==True:
                    forehead[i][j]=[255,255,255]
                if y_kmeans[i*cols+j]==False:
                    forehead[i][j]=[0,0,0]
        #Steps to get the length of forehead
        #1.get midpoint of the forehead
        #2.travel left side and right side
        #the idea here is to detect the corners of forehead which is the hair.
        #3.Consider the point which has change in pixel value (which is hair)
        forehead_mid = [int(cols/2), int(rows/2) ] #midpoint of forehead
        lef=0 
        #gets the value of forehead point
        pixel_value = forehead[forehead_mid[1],forehead_mid[0] ]
        for i in range(0,cols):
            #enters if when change in pixel color is detected
            if forehead[forehead_mid[1],forehead_mid[0]-i].all()!=pixel_value.all():
                lef=forehead_mid[0]-i
                break;
        left = [lef,forehead_mid[1]]
        rig=0
        for i in range(0,cols):
            #enters if when change in pixel color is detected
            if forehead[forehead_mid[1],forehead_mid[0]+i].all()!=pixel_value.all():
                rig = forehead_mid[0]+i
                break;
        right = [rig,forehead_mid[1]]
        
        #drawing line1 on forehead with circles
        #specific landmarks are used. 

        line1 = np.subtract(right+y,left+x)[0]
        cv2.line(results, tuple(x+left), tuple(y+right), color=(0,255,0), thickness = 2)
        cv2.putText(results,' Line 1',tuple(x+left),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
        cv2.circle(results, tuple(x+left), 5, color=(255,0,0), thickness=-1)    
        cv2.circle(results, tuple(y+right), 5, color=(255,0,0), thickness=-1)        

        #drawing line 2 with circles
        linepointleft = (landmarks[1,0],landmarks[1,1])
        linepointright = (landmarks[15,0],landmarks[15,1])
        line2 = np.subtract(linepointright,linepointleft)[0]
        cv2.line(results, linepointleft,linepointright,color=(0,255,0), thickness = 2)
        cv2.putText(results,' Line 2',linepointleft,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255,0,0), thickness=-1)    
        cv2.circle(results, linepointright, 5, color=(255,0,0), thickness=-1)    

        #drawing line 3 with circles
        linepointleft = (landmarks[3,0],landmarks[3,1])
        linepointright = (landmarks[13,0],landmarks[13,1])
        line3 = np.subtract(linepointright,linepointleft)[0]
        cv2.line(results, linepointleft,linepointright,color=(0,255,0), thickness = 2)
        cv2.putText(results,' Line 3',linepointleft,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255,0,0), thickness=-1)    
        cv2.circle(results, linepointright, 5, color=(255,0,0), thickness=-1)    

        #drawing line 4 with circles
        linepointbottom = (landmarks[8,0],landmarks[8,1])
        linepointtop = (landmarks[8,0],y)
        line4 = np.subtract(linepointbottom,linepointtop)[1]
        cv2.line(results,linepointtop,linepointbottom,color=(0,255,0), thickness = 2)
        cv2.putText(results,' Line 4',linepointbottom,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
        cv2.circle(results, linepointtop, 5, color=(255,0,0), thickness=-1)    
        cv2.circle(results, linepointbottom, 5, color=(255,0,0), thickness=-1)    
        #print(line1,line2,line3,line4)

        similarity = np.std([line1,line2,line3])
        #print("similarity=",similarity)
        ovalsimilarity = np.std([line2,line4])
        #print('diam=',ovalsimilarity)

        #we use arcustangens for angle calculation
        ax,ay = landmarks[3,0],landmarks[3,1]
        bx,by = landmarks[4,0],landmarks[4,1]
        cx,cy = landmarks[5,0],landmarks[5,1]
        dx,dy = landmarks[6,0],landmarks[6,1]
        import math
        from math import degrees
        alpha0 = math.atan2(cy-ay,cx-ax)
        alpha1 = math.atan2(dy-by,dx-bx)
        alpha = alpha1-alpha0
        angle = abs(degrees(alpha))
        angle = 180-angle

        for i in range(1):
            if similarity<10:
                if angle<160:
                    # print('squared shape.Jawlines are more angular')
                    shape = 'squared'
                    break
                else:
                    # print('round shape.Jawlines are not that angular')
                    shape = 'round'
                    break
            if line3>line1:
                if angle<160:
                    # print('triangle shape.Forehead is more wider') 
                    shape = 'triangle'
                    break
            if ovalsimilarity<10:
                    # print('diamond shape. line2 & line4 are similar and line2 is slightly larger')
                    shape = 'diamond'
                    break
            if line4 > line2:
                if angle<160:
                    # print('rectangular. face length is largest and jawline are angular ')
                    shape = 'rectangular'
                    break;
                else:
                    # print('oblong. face length is largest and jawlines are not angular')
                    shape = 'oblong'
                    break;
            # print("Damn! Contact the developer")
        
        # output = np.concatenate((original,results), axis=1)
        # cv2.imshow('output',output)
        return shape

    def detect_skin_color(self):
        landmarks = self.landmarks
        # for x in np.linspace(landmarks.part(2).x, landmarks.part(16).x)
        image = self.face
        y1 = int((landmarks.part(40).y*2 + landmarks.part(5).y)/3)
        x1 = int((landmarks.part(1).x + landmarks.part(30).x)/2) 

        y2 = int((landmarks.part(47).y*2 + landmarks.part(10).y)/3)
        x2 = int((landmarks.part(15).x + landmarks.part(30).x)/2)
        
        # cv2.circle(image, (x1, y1), 4, (255, 0, 0), 1)
        # cv2.circle(image, (x2, y2), 4, (255, 0, 0), 1)

        # cv2.imshow('face', image)
        # cv2.waitKey(0)

        c1 = np.mean(np.mean(image[y1-2:y1+2, x1-2:x1+2, :], axis=0), axis=0)
        c2 = np.mean(np.mean(image[y2-2:y2+2, x2-2:x2+2, :], axis=0), axis=0)
        bgr_color = np.round(np.mean([c1, c2], axis=0)).astype(int)
        rgb_color = np.flip(bgr_color, axis=0)
        return rgb_color


    def detect_eye_color(self):
        # Right eye
        landmarks = self.landmarks
        image = self.face
        r_x_start = min([landmarks.part(43).x, landmarks.part(47).x])
        r_x_end = max([landmarks.part(44).x, landmarks.part(46).x])
        r_y_start = min([landmarks.part(43).y, landmarks.part(44).y])
        r_y_end = max([landmarks.part(47).y, landmarks.part(46).y])
        p = 0
        r_box = image[r_y_start-p:r_y_end+p, r_x_start-p:r_x_end+p]
        B = r_box[:,:,0]
        G = r_box[:,:,1]
        R = r_box[:,:,2]
        mask = (B < 220) & (G < 220) & (R < 220)
        B = B[mask]
        G = G[mask]
        R = R[mask]
        r_color = [R.mean(), G.mean(), B.mean()]

        # Left eye
        l_x_start = min([landmarks.part(37).x, landmarks.part(41).x])
        l_x_end = max([landmarks.part(38).x, landmarks.part(40).x])
        l_y_start = min([landmarks.part(37).y, landmarks.part(38).y])
        l_y_end = max([landmarks.part(41).y, landmarks.part(40).y])
        p = 0
        l_box = image[l_y_start-p:l_y_end+p, l_x_start-p:l_x_end+p]
        B = l_box[:,:,0]
        G = l_box[:,:,1]
        R = l_box[:,:,2]
        mask = (B < 200) & (G < 200) & (R < 200)
        B = B[mask]
        G = G[mask]
        R = R[mask]
        l_color = [R.mean(), G.mean(), B.mean()]

        eye_color = np.round(np.mean([r_color, l_color], axis=0)).astype(int)

        return eye_color



class Faces():
    def __init__(self, image, mode='dlib'):
        self.image = image
        self.mode = mode

    def detect_all(self):
        if self.mode == 'dlib':
            detector = dlib.get_frontal_face_detector()
            faces = detector(self.image)
            padding = 1
            result = []
            for face in faces:
                face_cropped = self.image[face.top()-padding:face.bottom()+padding, face.left()-padding:face.right()+padding]
                result.append(Face(face_cropped))

        else:
            raise Exception ('face detector mode is unknown')

        return result
