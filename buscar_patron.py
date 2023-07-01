import cv2
import numpy as np


class FindTemplate:

    def __init__(self):

        self.imagen = None
        self.points = []

    def captura_foto(self, ruta):
        """
        Activa la webcam y pulsando Q en el teclado realiza la foto
        :return:
        """

        # Crea un objeto VideoCapture. 0 significa que estás usando la cámara web integrada.
        cap = cv2.VideoCapture(0)

        # Verifica si la cámara web se abrió correctamente
        if not cap.isOpened():
            print("No se pudo abrir la cámara web.")
            exit()

        while True:
            # Captura fotograma por fotograma
            ret, frame = cap.read()

            # Si el fotograma se leyó correctamente, ret es True
            if not ret:
                break

            # Muestra el fotograma resultante
            cv2.imshow("webcam", frame)

            # Si presionas 'q' en tu teclado, saldrás del bucle y guardas una captura
            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite(ruta, frame)
                imagen = cv2.imread(ruta)
                break

        cap.release()
        cv2.destroyAllWindows()

        return imagen

    def dibujar_rectangulo(self, event, x, y, flags, param):

        # Si se ha hecho clic con el botón izquierdo, guarda el punto inicial
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) == 0:  # Solo recoge el primer punto si no hay puntos existentes
                self.points.append((x, y))

        # Si se ha liberado el botón izquierdo, guarda el punto final
        elif event == cv2.EVENT_LBUTTONUP:
            if len(self.points) == 1:  # Solo recoge el segundo punto si ya hay un punto existente
                self.points.append((x, y))
                cv2.rectangle(self.imagen, self.points[0], self.points[1], (0, 255, 0), 1)
                cv2.imshow("plantilla", self.imagen)

    def crear_plantilla(self, ruta):

        self.imagen = cv2.imread("imagenes/referencia.jpg")
        cv2.namedWindow("plantilla")
        cv2.setMouseCallback("plantilla", self.dibujar_rectangulo)
        while True:
            cv2.imshow("plantilla", self.imagen)
            if cv2.waitKey(1) == ord('q') or len(self.points) == 2:
                break

        if len(self.points) == 2:
            recorte = self.imagen[self.points[0][1]:self.points[1][1], self.points[0][0]:self.points[1][0]]
            cv2.imwrite(ruta, recorte)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return recorte

    def buscar_plantilla(self, template, method=cv2.TM_SQDIFF_NORMED, grayscale=True, normalize=True):

        imagen = self.captura_foto(ruta="imagenes/resultado.jpg")
        imagen_original = imagen.copy()

        # Convertir la imagen y el patrón a escala de grises si grayscale=True
        if grayscale:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            patron = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Normalizar la imagen y el patrón si normalize=True
        if normalize:
            cv2.normalize(imagen, imagen, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.normalize(template, template, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Recogemos altura y ancho de la imagen
        h, w = patron.shape[:2]

        # Creación de una máscara del mismo tamaño que la plantilla
        mascara = np.zeros_like(patron)
        mascara[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255

        res = cv2.matchTemplate(image=imagen,
                                templ=patron,
                                method=method,
                                mask=mascara)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        esquina_sup_izq = min_loc
        esquina_inf_der = (esquina_sup_izq[0] + h, esquina_sup_izq[1] + w)
        print(f"min_val: {min_val}")
        print(f"max_val: {max_val}")

        # Imprimir las posiciones de las esquinas
        print(f"esquina_sup_izq: {esquina_sup_izq}")
        print(f"esquina_inf_der: {esquina_inf_der}")

        cv2.rectangle(imagen_original,
                      esquina_sup_izq,
                      esquina_inf_der,
                      (255, 255, 0),
                      2)
        cv2.imwrite("imagenes/resultado.jpg", imagen_original)

        return imagen_original

    def mostrar_resultados(self, template):

        imagen = self.buscar_plantilla(template=template)
        cv2.imshow("Resultado", imagen)
        cv2.waitKey()
        cv2.destroyAllWindows()
