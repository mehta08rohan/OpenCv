import cv2
import numpy as np
import math
import pyautogui

def read_show_resize_img():
    img = cv2.imread('sachin.png',1)
    resized_img = cv2.resize(img,(int(img.shape[1]*2),int(img.shape[0]*2)))
    cv2.imshow('abc',resized_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def face_detection(pic):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img= cv2.imread(pic)
    img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,1.1,4)


    for (x,y,h,w) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow('img',img)
    cv2.waitKey()


def detect_face_in_video():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)
    while True:
        _ , img = video.read()
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img,1.1,4)

        for (x,y,h,w) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        cv2.imshow('img', img)
    
        k = cv2.waitKey(30) & 0xff
        
        if k in [13,27]:
            break

    video.release()


def detect_hand_gestures():
    hand_cascade = cv2.CascadeClassifier('hand.xml')
    video = cv2.VideoCapture(0)
    
    count=0
    while True:
        _,frame = video.read()
        # frame = cv2.flip(frame,1)
        gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hands = hand_cascade.detectMultiScale(gray_img,1.5,2)
        contour = hands
        contour = np.array(contour)
        
        # print("count" , count)
        # print("contour" , len(contour))
        if count==0: 
  
            if len(contour)==2: 

                # pyautogui.write('Hello')
                cv2.putText(img=frame, text='Your engine started', 
                            org=(int(100 / 2 - 20), int(100 / 2)), 
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,  
                            fontScale=1, color=(0, 255, 0)) 
                            
                for (x, y, w, h) in hands: 
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                count += 1
  
        if count>0: 
            
            if len(contour)>=10 or len(contour)==1 : 

                pyautogui.press('space')
                cv2.putText(img=frame, text='Closed Fist', 
                            org=(int(100 / 2 - 20), int(100 / 2)), 
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,  
                            fontScale=1, color=(255, 0, 0)) 

                            
                for (x, y, w, h) in hands: 
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
              
  
            # elif len(contour)==1: 
            #     cv2.putText(img=frame, text='You can speed upto 80km/h', 
            #             org=(int(100 / 2 - 20), int(100 / 2)), 
            #             fontFace=cv2.FONT_HERSHEY_DUPLEX,  
            #             fontScale=1, color=(0, 255, 0)) 
                          
                # for (x, y, w, h) in hands: 
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            #Fist
            elif len(contour)==0: 
                
                cv2.putText(img=frame, text='Open Fist', 
                        org=(int(100 / 2 - 20), int(100 / 2)), 
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,  
                        fontScale=1, color=(0, 0, 255)) 
  
            count += 1
  
        cv2.imshow('T_Rex', frame) 
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break




# def detect_hand_gestures():
#     capture = cv2.VideoCapture(0)

#     while capture.isOpened():

#         ret, frame = capture.read()

#         cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
#         crop_image = frame[100:300, 100:300]
#         blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
#         hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        
#         mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

        
#         kernel = np.ones((5, 5))

        
#         dilation = cv2.dilate(mask2, kernel, iterations=1)
#         erosion = cv2.erode(dilation, kernel, iterations=1)

        
#         filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
#         ret, thresh = cv2.threshold(filtered, 127, 255, 0)

        
#         cv2.imshow("Thresholded", thresh)

        
#         contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#         try:
            
#             contour = max(contours, key=lambda x: cv2.contourArea(x))

            
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

            
#             hull = cv2.convexHull(contour)

#             # Draw contour
#             drawing = np.zeros(crop_image.shape, np.uint8)
#             cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
#             cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

#             # Find convexity defects
#             hull = cv2.convexHull(contour, returnPoints=False)
#             defects = cv2.convexityDefects(contour, hull)

            
#             count_defects = 0

#             for i in range(defects.shape[0]):
#                 s, e, f, d = defects[i, 0]
#                 start = tuple(contour[s][0])
#                 end = tuple(contour[e][0])
#                 far = tuple(contour[f][0])

#                 a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#                 b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#                 c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#                 angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                
#                 if angle <= 90:
#                     count_defects += 1
#                     cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

#                 cv2.line(crop_image, start, end, [0, 255, 0], 2)

            
#             if count_defects == 0:
#                 cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
#             elif count_defects == 1:
#                 cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
#             elif count_defects == 2:
#                 cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
#             elif count_defects == 3:
#                 cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
#             elif count_defects == 4:
#                 cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
#             else:
#                 pass
#         except:
#             pass

        
#         cv2.imshow("Gesture", frame)
#         all_image = np.hstack((drawing, crop_image))
#         cv2.imshow('Contours', all_image)

        
#         if cv2.waitKey(1) == ord('q'):
#             break







    



        

        



# detect_face_in_video()
# face_detection('a.jpg')
# read_show_resize_img()
detect_hand_gestures()