# parpadeo.py
import cv2
import numpy as np
from scipy.spatial import distance as dist
import time

class BlinkDetector:
    def __init__(self):
        # Constantes
        self.EYE_AR_THRESH = 0.20
        self.EYE_AR_CONSEC_FRAMES = 50
        
        # Variables de estado
        self.counter = 0
        self.total_blinks = 0
        self.alarm_on = False
        self.blink_start = 0
        self.sleep_time = 0
        
        # Puntos de los ojos
        self.LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]

    def calculate_ear(self, eye_points):
        """Calcula la relación de aspecto del ojo (EAR)"""
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C) if C > 0 else 0

    def process_frame(self, frame, face_landmarks, elapsed_time):
        """Procesa un frame para detectar parpadeos y somnolencia"""
        # Obtener coordenadas de los ojos
        left_eye = []
        right_eye = []

        # Extraer puntos del ojo izquierdo
        for left_point in self.LEFT_EYE_POINTS:
            x = int(face_landmarks.landmark[left_point].x * frame.shape[1])
            y = int(face_landmarks.landmark[left_point].y * frame.shape[0])
            left_eye.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Extraer puntos del ojo derecho
        for right_point in self.RIGHT_EYE_POINTS:
            x = int(face_landmarks.landmark[right_point].x * frame.shape[1])
            y = int(face_landmarks.landmark[right_point].y * frame.shape[0])
            right_eye.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Dibujar contornos de los ojos
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)

        # Calcular EAR
        left_ear = self.calculate_ear(np.array(left_eye))
        right_ear = self.calculate_ear(np.array(right_eye))
        ear = (left_ear + right_ear) / 2.0

        # Detectar parpadeo/somnolencia
        if ear < self.EYE_AR_THRESH:
            self.counter += 1
            if self.blink_start == 0:
                self.blink_start = time.time()

            if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                self.sleep_time = time.time() - self.blink_start
                if not self.alarm_on:
                    self.alarm_on = True
                cv2.putText(
                    frame, 
                    "DORMIDO", 
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    (0, 0, 255), 
                    3
                )
                cv2.putText(
                    frame, 
                    f"Tiempo: {self.sleep_time:.1f}s", 
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 255), 
                    2
                )
        else:
            if self.counter >= 3:
                self.total_blinks += 1
            self.counter = 0
            self.alarm_on = False
            self.blink_start = 0
            cv2.putText(
                frame, 
                "DESPIERTO", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )

        # Mostrar información en el frame
        if elapsed_time > 0:
            blinks_per_minute = (self.total_blinks / elapsed_time) * 60
            cv2.putText(
                frame, 
                f"Parpadeos/min: {blinks_per_minute:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 111, 0), 
                2
            )

        ear_text = f"EAR: {ear:.2f}"
        (text_width, _), _ = cv2.getTextSize(
            ear_text, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            2
        )
        cv2.putText(
            frame, 
            ear_text,
            (frame.shape[1] - text_width - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 191, 0), 
            2
        )

        cv2.putText(
            frame, 
            f"Parpadeos: {self.total_blinks}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )

        return frame