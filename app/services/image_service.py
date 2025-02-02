import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

class ImageService:

    #1. CARGA Y LECTURA DE IMAGENES
    @staticmethod
    async def display_image(image):
        _, img_encoded = cv2.imencode('.png', image)
        return img_encoded.tobytes()
    
    @staticmethod
    async def image_shape(image):
        return image.shape
    
    #2. SEGMENTACION DE IMAGENES
    @staticmethod
    async def segment_by_color(image, color):
        # Convertir imagen a espacio de color HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definir los rangos de color según el color elegido
        if color == 'red':
            lower_range = np.array([0, 50, 50])
            upper_range = np.array([10, 255, 255])
            lower_range2 = np.array([170, 50, 50])
            upper_range2 = np.array([180, 255, 255])
        elif color == 'green':
            lower_range = np.array([40, 50, 50])
            upper_range = np.array([90, 255, 255])
        elif color == 'blue':
            lower_range = np.array([90, 50, 50])
            upper_range = np.array([130, 255, 255])
        else:
            print("Color no reconocido.")
            return None
        
        # Crear máscara de segmentación usando inRange
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        
        if color == 'red':
            mask2 = cv2.inRange(hsv_image, lower_range2, upper_range2)
            mask = cv2.bitwise_or(mask, mask2)
        
        # Aplicar la máscara a la imagen original
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        _, img_encoded = cv2.imencode('.png', segmented_image)
        return img_encoded.tobytes()
    
    @staticmethod
    async def segment_by_kmeans(image, k_clusters):
        # Convertir la imagen a formato flotante y a una matriz de datos
        img_data = image.reshape((-1, 3))
        img_data = np.float32(img_data)
        
        # Definir criterios de parada para K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Aplicar K-means
        _, labels, centers = cv2.kmeans(img_data, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convertir los centros de los clusters a formato uint8 y obtener la imagen segmentada
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        _, img_encoded = cv2.imencode('.png', segmented_image)
        return img_encoded.tobytes()
    
    #3. CONVERSION Y AJUSTES DE IMAGENES
    @staticmethod
    async def adjust_brightness(image, brightness):
        brightness = np.clip(brightness, -100, 100)
        factor = 1 + (brightness / 100)
        image = image.astype(np.float32)
        image = image * factor
        image = np.clip(image, 0, 255)
        image_brightness = image.astype(np.uint8)
        _, img_encoded = cv2.imencode('.png', image_brightness)
        return img_encoded.tobytes()

    @staticmethod
    async def adjust_contrast(image, contrast):
        contrast = np.clip(contrast, -100, 100)
        factor = 1 + (contrast / 100)
        image = image.astype(np.float32)
        image = factor * (image - 128) + 128
        image = np.clip(image, 0, 255)
        image_contrast = image.astype(np.uint8)
        _, img_encoded = cv2.imencode('.png', image_contrast)
        return img_encoded.tobytes()

    @staticmethod
    async def adjust_shadows(image, shadow):
        factor = (shadow + 100) / 100
        shadows_mask = image < 128
        adjusted_image = np.where(shadows_mask, np.clip(image * factor, 0, 255), image).astype(np.uint8)
        _, img_encoded = cv2.imencode('.png', adjusted_image)
        return img_encoded.tobytes()

    @staticmethod
    async def adjust_exposure(image, exposure):
        factor = (exposure + 100) / 100
        adjusted_image = np.clip(image * factor, 0, 255).astype(np.uint8)
        _, img_encoded = cv2.imencode('.png', adjusted_image)
        return img_encoded.tobytes()
    
    @staticmethod
    async def adjust_highlights(image, highlight_value):
        highlight_factor = (highlight_value + 100) / 100
        highlights_mask = image >= 128
        adjusted_image = np.where(highlights_mask, np.clip(image * highlight_factor, 0, 255), image).astype(np.uint8)
        _, img_encoded = cv2.imencode('.png', adjusted_image)
        return img_encoded.tobytes()
    
    @staticmethod
    async def grayscale(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_encoded = cv2.imencode('.png', gray_image)
        return img_encoded.tobytes()
    
    #!CORREGIR ENVIO DE PARAMETROS
    @staticmethod
    async def grayscale_rgb(image, red_value, green_value, blue_value):
        gray_image = np.zeros_like(image)
        gray_image = np.dot(image, [blue_value, green_value, red_value])
        gray_image = np.clip(gray_image, 0, 255)
        gray_image = gray_image.astype(np.uint8)
        _, img_encoded = cv2.imencode('.png', gray_image)
        return img_encoded.tobytes()
    
    @staticmethod
    async def negative(image):
        negative_image = cv2.bitwise_not(image)
        _, img_encoded = cv2.imencode('.png', negative_image)
        return img_encoded.tobytes()
    
    #4. TRANSFORMACIONES Y ESCALADO DE IMAGENES
    @staticmethod
    async def scale_image(image, scale_x, scale_y):
        # Obtener dimensiones de la imagen original
        alto, ancho = image.shape[:2]
        
        # Calcular nuevos tamaños de imagen basados en las escalas
        nuevo_ancho = int(ancho * scale_x)
        nuevo_alto = int(alto * scale_y)
        
        # Aplicar escalado a la imagen
        imagen_escalada = cv2.resize(image, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
        _, img_encoded = cv2.imencode('.png', imagen_escalada)
        return img_encoded.tobytes()

#!NO TERMINADA
    @staticmethod
    async def rotate_image(image, angle):
        # Obtener dimensiones de la imagen original
        alto, ancho = image.shape[:2]
        
        # Calcular el centro de la imagen
        centro = (ancho // 2, alto // 2)
        
        # Definir la matriz de transformación de rotación
        matriz_rotacion = cv2.getRotationMatrix2D(centro, angle, 1.0)
        
        # Calcular el nuevo tamaño de la imagen para asegurarse de que la imagen rotada quede completamente visible
        cos = np.abs(matriz_rotacion[0, 0])
        sin = np.abs(matriz_rotacion[0, 1])
        
        nuevo_ancho = int((alto * sin) + (ancho * cos))
        nuevo_alto = int((alto * cos) + (ancho * sin))
        
        # Ajustar la matriz de transformación para tomar en cuenta el cambio de tamaño
        matriz_rotacion[0, 2] += (nuevo_ancho / 2) - centro[0]
        matriz_rotacion[1, 2] += (nuevo_alto / 2) - centro[1]
        
        # Aplicar la transformación de rotación a la imagen
        imagen_transformada = cv2.warpAffine(image, matriz_rotacion, (nuevo_ancho, nuevo_alto))
    
        _, img_encoded = cv2.imencode('.png', imagen_transformada)
        return img_encoded.tobytes()
    
    #5. ANALISIS Y ESTADISTICAS DE IMAGENES
    @staticmethod
    async def image_histogram(image):

        if len(image.shape) > 2:
            imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = image

        hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256])

        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Intensidad de píxel")
        plt.ylabel("Número de píxeles")
        plt.plot(hist)
        plt.xlim([0, 256])

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf.read()
    
    @staticmethod
    async def equalize_image(image):

        if len(image.shape) > 2:
            imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = image

        imagen_ecualizada = cv2.equalizeHist(imagen_gris)
        _, img_encoded = cv2.imencode('.png', imagen_ecualizada)
        return img_encoded.tobytes()
    
    async def equalize_histogram(image):
        if len(image.shape) > 2:
            imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = image

        imagen_ecualizada = cv2.equalizeHist(imagen_gris)

        # Calcular el histograma de la imagen ecualizada
        hist = cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Intensidad de píxel")
        plt.ylabel("Número de píxeles")
        plt.plot(hist)
        plt.xlim([0, 256])

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf.read()
    
    #6. DETECCION DE CARACTERISTICAS Y OBJETOS
    @staticmethod
    async def edge_detection(image, min_area):
        # Cargar la imagen utilizando la función load_image
        #img_gray = load_image(img_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Binarizar la imagen en escala de grises
        ret, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
    
        # Encontrar contornos en la imagen binarizada
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # Preparar una copia de la imagen original en color para dibujar contornos

        img_copy = image.copy()
    
        # Enumerar y enmarcar los objetos que cumplen con el área mínima
        object_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                object_count += 1
                perimeter = cv2.arcLength(cnt, True)
                epsilon = 0.0019 * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(img_copy, [approx], -1, (255, 0, 0), 3)
                cv2.putText(img_copy, f'Objeto {object_count}', tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
        # Mostrar la imagen con los contornos de los objetos detectados

        _, img_encoded = cv2.imencode('.png', img_copy)
        return img_encoded.tobytes()

    
    @staticmethod
    async def objects_detection(image, min_area):
        """
        Detecta objetos en una imagen y enmarca los que superan el área mínima especificada.
        Args:
        - image_path (str): Ruta de la imagen a procesar.
        - min_area (int): Área mínima requerida para enmarcar un objeto.
        Returns:
        - img_copy (numpy.ndarray): Imagen con los objetos enmarcados y etiquetados.
        """
        # Cargar la imagen en escala de grises
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Binarizar la imagen
        ret, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
        # Encontrar contornos en la imagen binarizada
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Copiar la imagen original para dibujar los contornos y etiquetas
        img_copy = image.copy()
        # Enumerar y enmarcar los objetos que cumplen con el área mínima
        object_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                object_count += 1
                # Calcular el perímetro
                perimeter = cv2.arcLength(cnt, True)
                # Calcular el centroide
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = 0, 0
                # Imprimir información del objeto detectado
                print(f"Objeto {object_count}:")
                print(f"  - Área: {area}")
                print(f"  - Perímetro: {perimeter}")
                print(f"  - Centroide: ({cx}, {cy})")
                # Enmarcar el objeto con un rectángulo
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (100, 100, 100), 3)
                # Dibujar el centroide
                cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
                # Etiquetar el objeto con su número
                cv2.putText(img_copy, str(object_count), (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
        _, img_encoded = cv2.imencode('.png', img_copy)
        return img_encoded.tobytes()
    
    #7. FILTROS Y MASCARAS
    #MASKS
    @staticmethod
    async def filter_color_mask(image, color, return_image):
        # Convertir la imagen a espacio de color HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definir rangos de color en HSV según el color especificado
        if color == 'blue':
            lower = np.array([100, 150, 50])
            upper = np.array([130, 255, 255])
        elif color == 'red':
            lower1 = np.array([0, 50, 50])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 50, 50])
            upper2 = np.array([180, 255, 255])
        elif color == 'green':
            lower = np.array([35, 50, 50])
            upper = np.array([85, 255, 255])
        else:
            raise ValueError("Color no soportado. Usa 'blue', 'red' o 'green'.")
        
        # Crear la máscara para el color
        if color == 'red':
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, lower, upper)
        
        # Aplicar la máscara para obtener la imagen resultante
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Devolver la imagen resultante o la máscara según el parámetro booleano
        if return_image:
            _, img_encoded = cv2.imencode('.png', result)
            return img_encoded.tobytes()
        else:
            _, img_encoded = cv2.imencode('.png', mask)
            return img_encoded.tobytes()
        
    #KERNELS
    @staticmethod
    async def blur_kernel(image):
        # Definir el kernel de desenfoque
        kernel_des = np.ones((5, 5), np.float32) / 25
        imagen_des = cv2.filter2D(image, -1, kernel_des)
        _, img_encoded = cv2.imencode('.png', imagen_des)
        return img_encoded.tobytes()
    
    @staticmethod
    async def sobel_kernel(image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Definir el kernel de detección de bordes Sobel
        kernel_sobel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

        # Aplicar el filtro
        imagen_bordes = cv2.filter2D(image, -1, kernel_sobel)
        _, img_encoded = cv2.imencode('.png', imagen_bordes)
        return img_encoded.tobytes()
    
    @staticmethod
    async def sharpen_kernel(image):
        # Definir el kernel de realce
        kernel_realce = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])

        # Aplicar el filtro
        imagen_realzada = cv2.filter2D(image, -1, kernel_realce)
        _, img_encoded = cv2.imencode('.png', imagen_realzada)
        return img_encoded.tobytes()
    
    @staticmethod
    async def laplacian_kernel(image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Definir el kernel de detección de bordes Laplacian
        kernel_laplacian = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])

        # Aplicar el filtro
        imagen_bordes_lap = cv2.filter2D(image, -1, kernel_laplacian)
        _, img_encoded = cv2.imencode('.png', imagen_bordes_lap)
        return img_encoded.tobytes()
    
    @staticmethod
    async def embossing_kernel(image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Definir el kernel de embossing
        kernel_emboss = np.array([[-2, -1, 0],
                                [-1, 1, 1],
                                [0, 1, 2]])

        # Aplicar el filtro
        imagen_emboss = cv2.filter2D(image, -1, kernel_emboss)
        _, img_encoded = cv2.imencode('.png', imagen_emboss)
        return img_encoded.tobytes()

    #FILTROS
    @staticmethod
    async def gaussian_blur_filter(image, kernel_size):
        result_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        _, img_encoded = cv2.imencode('.png', result_image)
        return img_encoded.tobytes()
    
    async def average_blur_filter(image, kernel_size):
        result_image = cv2.blur(image, (kernel_size, kernel_size))
        _, img_encoded = cv2.imencode('.png', result_image)
        return img_encoded.tobytes()
    
    #!CORREGIR PARA QUE SEA SIN CANNY
    async def edge_detection_filter(image):
        result_image = cv2.Canny(image, 100, 200)
        _, img_encoded = cv2.imencode('.png', result_image)
        return img_encoded.tobytes()
    
    async def laplace_filter(image):
        result_image = cv2.Laplacian(image, cv2.CV_64F)
        _, img_encoded = cv2.imencode('.png', result_image)
        return img_encoded.tobytes()
    
    async def sobel_filter(image, x_order, y_order):
        result_image = cv2.Sobel(image, cv2.CV_64F, x_order, y_order, ksize=5)
        _, img_encoded = cv2.imencode('.png', result_image)
        return img_encoded.tobytes()
    
    async def canny_filter(image):
        result_image = cv2.Canny(image, 100, 200)
        _, img_encoded = cv2.imencode('.png', result_image)
        return img_encoded.tobytes()
    #8. OPERACIONES CON IMAGENES
    @staticmethod
    async def manual_convolution(imagen, kernel_type):
        
        if kernel_type == "blur":
            kernel_blur = np.ones((5, 5), np.float32) / 25
            kernel = kernel_blur
        elif kernel_type == "sobel":
            kernel_sobel = np.array([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]])
            kernel = kernel_sobel
        elif kernel_type == "realce":
            kernel_realce = np.array([[0, -1, 0],
                                            [-1, 5, -1],
                                            [0, -1, 0]])
            kernel = kernel_realce
        elif kernel_type == "laplacian":
            kernel_laplacian = np.array([[0, 1, 0],
                                                [1, -4, 1],
                                                [0, 1, 0]])
            kernel = kernel_laplacian
        elif kernel_type == "embossing":
            kernel_embossing = np.array([[-2, -1, 0],
                                            [-1, 1, 1],
                                            [0, 1, 2]])
            kernel = kernel_embossing
        else:
            print("Kernel no encontrado")
            
        
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        altura, ancho = imagen.shape
        k_altura, k_ancho = kernel.shape
        
        # Calcular el padding requerido para mantener el tamaño de la imagen
        pad_altura = k_altura // 2
        pad_ancho = k_ancho // 2
        
        # Crear una imagen con padding
        imagen_padding = np.zeros((altura + 2 * pad_altura, ancho + 2 * pad_ancho))
        imagen_padding[pad_altura:pad_altura+altura, pad_ancho:pad_ancho+ancho] = imagen
        
        # Aplicar la convolución
        imagen_filtrada = np.zeros_like(imagen, dtype=np.float32)
        
        for i in range(altura):
            for j in range(ancho):
                imagen_filtrada[i, j] = np.sum(imagen_padding[i:i+k_altura, j:j+k_ancho] * kernel)
        
        result_image = imagen_filtrada.astype(np.uint8)
        _, img_encoded = cv2.imencode('.png', result_image)
        return img_encoded.tobytes()
    
    @staticmethod
    async def fourier_transform(image):
        # Convertir la imagen a escala de grises
        imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar la Transformada de Fourier
        f = np.fft.fft2(imagen_gris)
        fshift = np.fft.fftshift(f)
        
        # Calcular el espectro de magnitud
        magnitud_espectro = 20 * np.log(np.abs(fshift))
        
        # Normalizar la imagen del espectro para que los valores estén en el rango [0, 255]
        magnitud_espectro = cv2.normalize(magnitud_espectro, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convertir a tipo de datos uint8
        magnitud_espectro = np.uint8(magnitud_espectro)
        
        _, img_encoded = cv2.imencode('.png', magnitud_espectro)
        return img_encoded.tobytes()
    
    @staticmethod
    async def dilatation(image, iterations):
        kernel = np.ones((5, 5), np.uint8)
        image_result = cv2.dilate(image, kernel, iterations=iterations)
        _, img_encoded = cv2.imencode('.png', image_result)
        return img_encoded.tobytes()
    
    @staticmethod
    async def erosion(image, iterations):
        kernel = np.ones((5, 5), np.uint8)
        image_result = cv2.erode(image, kernel, iterations=iterations)
        _, img_encoded = cv2.imencode('.png', image_result)
        return img_encoded.tobytes()
    
    @staticmethod
    async def sum_images(image1, image2):
        image_result = cv2.add(image1, cv2.resize(image2, (image1.shape[1], image1.shape[0]),interpolation=cv2.INTER_AREA))
        _, img_encoded = cv2.imencode('.png', image_result)
        return img_encoded.tobytes()
    
    @staticmethod
    async def subtract_images(image1, image2):
        image_result = cv2.subtract(image1, cv2.resize(image2, (image1.shape[1], image1.shape[0]),interpolation=cv2.INTER_AREA))
        _, img_encoded = cv2.imencode('.png', image_result)
        return img_encoded.tobytes()
    #9. APLICACIONES EN TIEMPO REAL
