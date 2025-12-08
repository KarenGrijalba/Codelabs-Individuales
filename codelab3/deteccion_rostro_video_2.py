import cv2, dlib, time

# Ejercio con los cambios del punto 4.3

# Detector HOG de dlib
detector = dlib.get_frontal_face_detector()

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Estadísticas
frame_limit = 200
count = 0

# Contadores para comparación
fps_ups0 = []
fps_ups1 = []
det_ups0 = 0
det_ups1 = 0

print("=== Iniciando prueba con cámara ===")
print("Se evaluarán 200 frames...\n")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #       UPSAMPLE = 0

    t0 = time.time()
    rects0 = detector(gray, 0)
    t1 = time.time()

    fps_ups0.append(1 / (t1 - t0))
    det_ups0 += len(rects0)

    # -------------------------
    #       UPSAMPLE = 1
    # -------------------------
    t2 = time.time()
    rects1 = detector(gray, 1)
    t3 = time.time()

    fps_ups1.append(1 / (t3 - t2))
    det_ups1 += len(rects1)

    # Dibujar detecciones (solo UPSAMPLE=1 para visualizar)
    for r in rects1:
        x, y, w, h = r.left(), r.top(), r.width(), r.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar ventana
    cv2.imshow("Dlib HOG (UPSAMPLE=1)", frame)

    count += 1

    # salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # terminar al llegar a 200 frames
    if count >= frame_limit:
        break

cap.release()
cv2.destroyAllWindows()

#    RESULTADOS FINALES

print("\n=== RESULTADOS ===")

print(f"UPSAMPLE = 0")
print(f"- FPS promedio: {sum(fps_ups0)/len(fps_ups0):.2f}")
print(f"- Total caras detectadas: {det_ups0}")

print("\nUPSAMPLE = 1")
print(f"- FPS promedio: {sum(fps_ups1)/len(fps_ups1):.2f}")
print(f"- Total caras detectadas: {det_ups1}")
