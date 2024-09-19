import cv2
import numpy as np
from time import sleep

largura_min=80 
#minimum width of rectangle
altura_min=80
 # minimum height of the retangle

offset=6 
#Allowed error in pixels  

pos_linha=550
 #Position of the counting lines

delay= 60 
#indicating vídeo has 60frames per second 
#variables
detec = []
carros= 0

#functions to calculate the center of rectangle	
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy
#open video file
cap = cv2.VideoCapture('video.mp4')
subtracao = cv2.createBackgroundSubtractorMOG2()

while True:
    ret , frame1 = cap.read()
    #read a frame from the video
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    #convert frame from blur to grayscale
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    #dilate the foreground mask 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)#close small gaps
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255,127,0), 3) 
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2) 
           #draw the rectangle around the detected vehicle    
        centro = pega_centro(x, y, w, h)
        # gets the centre of rectangle
        detec.append(centro) 
        #adding centre point  to the detection list
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
            #checks if centre point croses the counting line
            if y<(pos_linha+offset) and y>(pos_linha-offset):
                carros+=1   
                #increment the vehicle counter
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0,127,255), 3)  
                detec.remove((x,y))
                print("vehicle is detected : "+str(carros))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilatada)
      #show the processed fram for detection

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
 #Release the video capture object and close all openCV windows
cap.release()
