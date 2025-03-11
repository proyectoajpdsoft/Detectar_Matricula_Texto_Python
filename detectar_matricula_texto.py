from os.path import exists
import cv2
import matplotlib.pyplot as plt 
import imutils
import numpy as np
import easyocr

ficheroImagen = "D:\\Mis documentos\\ProyectoA\\Python\\matricula\\matricula.jpg"

if exists(ficheroImagen):
    # Leemos la imagen con cv2
    imagen = cv2.imread(ficheroImagen)
    # Por si queremos mostrar la imagen leída en una ventana
    #plt.imshow(imagen)
    #plt.show()

    # Transformamos la imagen a escala de grises
    imagenEscalaGrises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Por si queremos mostrar la imagen
    #plt.imshow(gray, cmap='gray')
    #plt.show()

    # Aplicamos filtro bilateral (limpia la imagen de posibles ruidos)
    # Esto hará que el paso de detección de bordes sea más preciso
    # Los parámetros 11, 17, 17 determinan el diámetro de la vecindad de píxeles, sigmaColor y sigmaSpace respectivamente
    imagenLimpia = cv2.bilateralFilter(imagenEscalaGrises, 11, 17, 17)
    # Por si queremos mostrar la imagen
    #plt.imshow(imagenLimpia, cmap='gray')
    #plt.show()

    # Detectar contornos de la imagen
    # Busca en la foto lugares donde el brillo o el color cambian bruscamente
    imagenContornos = cv2.Canny(imagenLimpia, 30, 200)
    # Por si queremos mostrar la imagen
    #plt.imshow(imagenContornos, cmap='gray')
    #plt.show()

    # Detectar curvas de nivel
    # Son los límites de los componentes conectados en la imagen
    imagenCurvasNivel = cv2.findContours(imagenContornos.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print (imagenCurvasNivel)

    # Ordenamos los contornos en función de su área
    # Se seleccionan los 5 contornos más grandes
    contornosMasGrandes = imutils.grab_contours(imagenCurvasNivel)
    contornosMasGrandes = sorted(contornosMasGrandes, key=cv2.contourArea, reverse=True)[:5]
    #print (contornosMasGrandes)
    
    # Esto no es necesario, solo se muestran los contornos obtenidos para depurar
    # Muestra los contornos en la imagen original
    #imagenContornosDibujados = cv2.drawContours(imagen, contornosMasGrandes, -1, (0, 255, 0), 3)
    #plt.imshow(imagenContornosDibujados, cmap='gray')
    #plt.show()

    # La matrícula tendrá una forma rectangular, por lo que recorreremos los contornos obtenidos
    # para seleccionar el contorno que se parezca a un rectángulo (4 puntos)
    rectanguloLocalizado = []
    for contorno in contornosMasGrandes:
        
        # PARA CALCULAR epsilon
        # epsilon es el parámetro que especifica la precisión de aproximación
        # Mediante cv2.arcLength calcularemos el perímetro del contorno
        # Para la función cv2.arcLength, estableeremos los parámetros contorno y true si porque la figura es cerrada
        # Se multiplicará por cierto porcentaje para obtener epsilon
        #epsilon = 0.01 * cv2.arcLength(contorno, True)
        #contornoAproximado = cv2.approxPolyDP(contorno, epsilon, True)
        
        # Para usar un epsilon fijo: 10
        contornoAproximado = cv2.approxPolyDP(contorno, 10, True)
        
        # No neceario, solo para mostrar el contorno elegido en la imagen, para depurar 
        """
        mascara = np.zeros(imagenEscalaGrises.shape, np.uint8)
        contornoMatricula = cv2.drawContours(mascara, [contornoAproximado], 0, 255, -1)
        contornoMatricula = cv2.bitwise_and(imagen, imagen, mask=mascara)
        plt.imshow(contornoMatricula, cmap='gray')
        plt.show()
        """
        
        # Si el contorno aproximado tiene 4 "vértices" suponemos que es la matrícula
        if len(contornoAproximado) == 4:
            rectanguloLocalizado = contornoAproximado
            break
    
    #print(rectanguloLocalizado)

    # Si se ha encontrado una forma rectangular de cuatro puntos
    if len(rectanguloLocalizado) > 0:
        # Se crea una máscara de la misma forma que la imagen en escala de grises, rellena con ceros
        # El contorno de la matrícula encontrada se dibuja en esta máscara
        # Con la máscara, la matrícula real se extrae de la imagen original
        mascara = np.zeros(imagenEscalaGrises.shape, np.uint8)
        contornoMatricula = cv2.drawContours(mascara, [rectanguloLocalizado], 0, 255, -1)
        #plt.imshow(contornoMatricula, cmap='gray')
        #plt.show()        
        contornoMatricula = cv2.bitwise_and(imagen, imagen, mask=mascara)
        #plt.imshow(contornoMatricula, cmap='gray')
        #plt.show()

        # Con la máscara, se determinan las coordenadas delimitadoras de la matrícula 
        # Con estas coordenadas, la matrícula se extrae de la imagen en escala de grises
        (x, y) = np.where(mascara == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        matriculaFinal = imagenEscalaGrises[x1:x2 + 1, y1:y2 + 1]
        #plt.imshow(matriculaFinal, cmap='gray')
        #plt.show()

        # Usaremos el OCR previamente entrenado easyocr (se le pasa el idima)       
        leerOCR = easyocr.Reader(['es'], gpu=False)
        textoReconocido = leerOCR.readtext(matriculaFinal)
        # Imprimimos por pantalla el texto reconocido y sus datos (coordenadas de posición, probabilidad, etc.)
        # print(textoReconocido)
        
        # Para mostrar por pantalla sólo la matrícula obtenida
        matriculaObtenida = ""
        for (puntos, matricula, probabilidad) in textoReconocido:
            # print(f'Probabilidad: {probabilidad}')
            # print(f'Puntos del recuadro: {puntos}')
            matriculaObtenida = matriculaObtenida + matricula
        print(f'Matrícula: {matriculaObtenida}')
    else:
        print("No se han encontrado una forma rectangular en la imagen que tenga las características de una matrícula.")
else:
    print("No se ha encontrado el fichero de la imagen.")