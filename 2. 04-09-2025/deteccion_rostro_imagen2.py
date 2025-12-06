import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time

# Lista de imágenes (ajusta rutas según tus archivos)
imagenes = {
    "Una persona": "grupo.jpg",
    "Grupo": "grupo4.jpg",
    "Rostros parciales": "grupo2.jpg"
}

detector = MTCNN()

def procesar_imagen(nombre, ruta, thr_list=[0.6, 0.75, 0.9, 0.95], blur=False):
    # Carga imagen
    img = cv2.imread(ruta)
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen: {ruta}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Promediar tiempo de inferencia
    tiempos = []
    for _ in range(5):
        t0 = time()
        res = detector.detect_faces(img_rgb)
        t1 = time()
        tiempos.append((t1 - t0) * 1000)
    print(f"\n--- {nombre} ---")
    print(f"Promedio tiempo detección: {np.mean(tiempos):.1f} ms (5 corridas)")
    print(f"Rostros detectados totales: {len(res)}")

    # Filtrar según umbral
    for thr in thr_list:
        filtrados = [r for r in res if r['confidence'] >= thr]
        print(f"  Con thr={thr:.2f} → {len(filtrados)} rostros")

    # Visualización (con landmarks)
    vis = img_rgb.copy()
    for r in res:
        x, y, w, h = r['box']
        # Dibujar caja
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        # Dibujar landmarks
        for (px,py) in r['keypoints'].values():
            cv2.circle(vis, (px,py), 3, (255,0,0), -1)

        # Blur opcional para privacidad
        """
        if blur:
            face_roi = vis[y:y+h, x:x+w]
            if face_roi.size > 0:
                face_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                vis[y:y+h, x:x+w] = face_roi
        """

    plt.imshow(vis)
    plt.title(nombre)
    plt.axis('off')
    plt.show()


# Ejecutar en todas las imágenes
for nombre, ruta in imagenes.items():
    procesar_imagen(nombre, ruta, blur=True)
