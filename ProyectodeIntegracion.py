import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import time
s='s'
while s=='s':   ###################Loop de Entrada y Ciclo#############Presionar s para continuar
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
######################################## Deteccion de movimiento #########################   
    #Toma la imagen de referencia
    ret,frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    bordes1 = cv2.Canny(gray1, 100, 200)
    sumafilas= sum(bordes1)
    totalfilas=sum(sumafilas)
    
    while True:
        ret,frame = cap.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.resize(gray,(150,150),interpolation= cv2.INTER_CUBIC)     
        bordes = cv2.Canny(gray, 100, 200)
        sumaFilas1 = sum(bordes)
        totalfilas1=sum(sumaFilas1)
        #print(totalfilas1)
        #cv2.imshow('Bordes2',bordes)
    
        diferencia = totalfilas - totalfilas1
        diferenciaim=cv2.subtract(gray,gray1)
        sumfile=sum(diferenciaim)
        sumafilex=sum(sumfile)
        cv2.imshow('Imagen',diferenciaim)
    
        if abs(diferencia) > 5000:
            break
        
        totalfilas = totalfilas1
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

################################Deteccion de Rostros #################################################
    import cv2 as cv
    
    def reconocimiento_facial(ret1,frame1):
        dataPath = 'Images1' #Cambia a la ruta donde hayas almacenado Data
        imagePaths = os.listdir(dataPath)
        print('imagePaths=',imagePaths)
        face_recognizer = cv.face.EigenFaceRecognizer_create()
        
        # Leyendo el modelo
        face_recognizer.read('modeloEigenFace.xml')
        
        faceClassif = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        who1='Desconocido'
        count=0
        while True:
            ret,frame = ret1,frame1
            if ret == False: break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            auxFrame = gray.copy()
            faces = faceClassif.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv.resize(rostro,(150,150),interpolation= cv.INTER_CUBIC)
                result = face_recognizer.predict(rostro)
                cv.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv.LINE_AA)
                
                # EigenFaces
                if result[1] < 5700:
                    cv.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
                    cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                    #print(imagePaths[result[0]])
                    who1=imagePaths[result[0]]
                else:
                    cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
                    cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                    #print('Desconocido')
                    who1='Desconocido'
                
            cv.imshow('Rostro encontrado..',frame)
            k = cv.waitKey(1)
            count = count + 1
            if k == 27 or count < 3:
                break
        return who1
    
#########################################Logica Difusa#################################################
    
    def histo(frame):
        img_gris = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img_gris],[0],None, [256], [0, 256])
        histn=hist[20:245]
        hist1 = np.array(histn)
        y=np.where(hist1==max(histn))
        valhis=y[0]+20
        cv2.imshow('Histograma',frame)
        #print(valhis)
        return(valhis)
    
    def contorno(frame):
        grises = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _,th =  cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY_INV)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    
        i=0
        for c in cnts:
            area = cv2.contourArea(c)
            if area>2500 and area<200000:
                print(area)
                i=i+1
                if i>16:
                    i=16
    
        cv2.drawContours(frame,cnts,-1,(255,0,0),2)
        cv2.imshow('Bordes',frame)
        return(i)
    
    def internal(frame):
        j=4
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        val = np.sum(faces)
        if val>0:
            j=2
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('Internal',frame)
        return(j)
    
    internas = ctrl.Antecedent (np.arange(0,6,1),'internas')
    bordes = ctrl.Antecedent (np.arange(0,19,1),'bordes')
    histograma = ctrl.Antecedent (np.arange(0,256,1),'histograma')
    
    #Definir la salida
    salida = ctrl.Consequent(np.arange(0,21,1),'salida')
    
    internas['si']=fuzz.trapmf(internas.universe,[0,1,2,3])
    internas['no']=fuzz.trapmf(internas.universe,[2,3,4,5])
    #internas.view()
    
    bordes['no_tiene']=fuzz.trimf(bordes.universe,[0,0,2])
    bordes['menos_10']=fuzz.trimf(bordes.universe,[1,4,8])
    bordes['mas_10']=fuzz.trimf(bordes.universe,[7,10,15])
    #bordes.view()
    
    histograma['oscura']=fuzz.trimf(histograma.universe,[0,0,50])
    histograma['medio_oscura']=fuzz.trimf(histograma.universe,[25,50,102])
    histograma['medio_tono']=fuzz.trimf(histograma.universe,[50,102,153])
    histograma['altas_luces']=fuzz.trimf(histograma.universe,[102,153,204])
    histograma['blancos']=fuzz.trimf(histograma.universe,[153,204,255])
    #histograma.view()
    
    #Definición de Funciones de membresía de salida
    salida['rostro']=fuzz.trimf(salida.universe,[0,6,12])
    salida['objeto']=fuzz.trimf(salida.universe,[6,12,18])
    salida['desconocido']=fuzz.trimf(salida.universe,[12,18,24])
    
    #Definir las reglas
    #Si (formas internas es si AND histograma es medio oscura Or histograma es oscura OR histograma es blancos OR histograma es medio tono AND Bordes es menos de 10 OR no tiene bordes entonces la salida es rostro.
    rule1 = ctrl.Rule(internas['si']&(histograma['medio_oscura']|histograma['oscura']|histograma['altas_luces']|histograma['medio_tono']|histograma['blancos'])&(bordes['no_tiene']|bordes['menos_10']),salida['rostro'])
    #Si (#Si (formas internas es si AND histograma es medio tono OR histograma es blancos OR histograma es altas luces AND Bordes es mas de 10 OR menos de 10 bordes entonces la salida es rostro.
    rule2 = ctrl.Rule((internas['no'])&(histograma['oscura']|histograma['medio_oscura']|histograma['medio_tono']|histograma['altas_luces']|histograma['blancos'])&(bordes['no_tiene']|bordes['menos_10']),salida['objeto'])
    #Si internas es no AND (Histograma es Oscura OR Blancos OR Altas Luces) entonces la salida es Desconocido.
    rule3 = ctrl.Rule(internas['no']&(bordes['no_tiene']|bordes['menos_10'])&(histograma['oscura']|histograma['medio_oscura']|histograma['blancos']),salida['desconocido'])
    
    #Introducir al sistema las reglas
    salida_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    salida1 = ctrl.ControlSystemSimulation(salida_ctrl)
    
    cap = cv2.VideoCapture(0)
    ret,frame=cap.read()
    y=histo(frame)
    y1=np.real(y[0])
    i= contorno(frame)
    j=internal(frame)
    print("Interno:",j)
    print("Bordes",i)
    print("Histograma:",y1)
    cap.release()
    #Introducir los valores de entrada de cada categoria
    salida1.input['internas'] = j
    salida1.input['bordes'] = i
    salida1.input['histograma']= y1
    
    salida1.compute()
    
    print(salida1.output['salida'])
    
    salida.view(sim=salida1)
    
    var=salida1.output['salida']
    if var<8:
        resultado=reconocimiento_facial(ret,frame)
        print('Rostro Detectado')
        print("Bienvenido:",resultado)
        print("Histograma:",y1)
        print("Bordes",i)
        print("Interno:",j)
        print("Regla Numero 1 Aplicada")
        
    elif var<13.9:
        print('Esto es un Objeto, No voy a hacer nada....') 
        print("Histograma:",y1)
        print("Bordes",i)
        print("Interno:",j)
        print("Regla Numero 2 Aplicada")
    else:           
        print('desconocido')
        print("Histograma:",y1)
        print("Bordes",i)
        print("Interno:",j)
        print("Regla Numero 3 Aplicada")
    
    plt.show()
    
#################################Bucle de Contunar##############################
    s=input("Deseas continuar? (s/n)")
    cv2.destroyAllWindows()
    
