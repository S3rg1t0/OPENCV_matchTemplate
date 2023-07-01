from buscar_patron import FindTemplate


if __name__ == "__main__":

    detector = FindTemplate()
    detector.captura_foto(ruta="imagenes/referencia.jpg")
    plantilla = detector.crear_plantilla(ruta="imagenes/recorte.jpg")
    detector.mostrar_resultados(template=plantilla)


