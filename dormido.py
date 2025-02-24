# dormido.py
import cv2
import mediapipe as mp
import numpy as np
import time
from bostezo import YawnDetector
from parpadeo import BlinkDetector

def main():
    # Inicializar Mediapipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Iniciar detectores
    yawn_detector = YawnDetector()
    blink_detector = BlinkDetector()

    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Convertir la imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen
        results = face_mesh.process(frame_rgb)
        
        # Crear una copia del frame para dibujar
        output_frame = frame.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Procesar bostezos
                output_frame = yawn_detector.process_frame(
                    output_frame, 
                    face_landmarks, 
                    elapsed_time
                )
                
                # Procesar parpadeos
                output_frame = blink_detector.process_frame(
                    output_frame, 
                    face_landmarks, 
                    elapsed_time
                )

        # Mostrar FPS
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            fps_text = f"FPS: {fps:.1f}"
            (text_width, text_height), _ = cv2.getTextSize(
                fps_text,
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                1
            )
            cv2.putText(
                output_frame, 
                fps_text,
                (output_frame.shape[1] - text_width - 10,
                 output_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (0, 0, 0), 
                1
            )

        # Mostrar el frame
        cv2.imshow("AnderSein- ¿Te estas durmiendo?", output_frame)

        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()