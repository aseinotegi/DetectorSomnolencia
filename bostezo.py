# bostezo.py
import cv2
import numpy as np
from scipy.spatial import distance as dist
import time

class YawnDetector:
    def __init__(self):
        # Constantes ajustadas
        self.MOUTH_AR_THRESH = 0.6  # Reducido para ser más sensible
        self.MIN_YAWN_DURATION = 1.5  # Reducido ligeramente
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308
        
        # Puntos adicionales para mejor detección
        self.UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        self.LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # Variables de estado
        self.total_yawns = 0
        self.yawning = False
        self.yawn_start_time = None
        self.current_mar = 0
        self.mar_history = []  # Histórico de MAR para promediado
        
        # Puntos completos para el contorno de los labios
        self.LIPS_POINTS = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 308, 415, 310, 311, 312, 13, 82, 81, 80,
            191, 78, 62, 76, 77, 146, 91, 181, 84, 17,
            314, 405, 321, 375, 291
        ]

    def calculate_mar(self, mouth_points):
        """Calcula la relación de aspecto de la boca (MAR) con múltiples puntos"""
        # Calcular múltiples distancias verticales
        vertical1 = dist.euclidean(mouth_points[0], mouth_points[1])
        vertical2 = dist.euclidean(
            (mouth_points[0][0], mouth_points[0][1] - 5),
            (mouth_points[1][0], mouth_points[1][1] + 5)
        )
        vertical = max(vertical1, vertical2)  # Usar la mayor distancia
        horizontal = dist.euclidean(mouth_points[2], mouth_points[3])
        
        mar = vertical / horizontal if horizontal > 0 else 0
        return mar

    def process_frame(self, frame, face_landmarks, elapsed_time):
        """Procesa un frame para detectar bostezos con criterios de duración"""
        current_time = time.time()
        
        # Extraer puntos de los labios
        lips_points = []
        upper_lip_y = []
        lower_lip_y = []
        
        for lip_point in self.LIPS_POINTS:
            x = int(face_landmarks.landmark[lip_point].x * frame.shape[1])
            y = int(face_landmarks.landmark[lip_point].y * frame.shape[0])
            lips_points.append((x, y))
            
            if lip_point in self.UPPER_LIP:
                upper_lip_y.append(y)
            if lip_point in self.LOWER_LIP:
                lower_lip_y.append(y)

        lips_points = np.array(lips_points, dtype=np.int32)
        
        # Calcular apertura vertical
        upper_mean = np.mean(upper_lip_y)
        lower_mean = np.mean(lower_lip_y)
        vertical_distance = abs(upper_mean - lower_mean)
        
        # Extraer puntos principales para MAR
        mar_points = []
        for point in [self.MOUTH_TOP, self.MOUTH_BOTTOM, self.MOUTH_LEFT, self.MOUTH_RIGHT]:
            x = int(face_landmarks.landmark[point].x * frame.shape[1])
            y = int(face_landmarks.landmark[point].y * frame.shape[0])
            mar_points.append((x, y))

        # Calcular MAR actual
        self.current_mar = self.calculate_mar(mar_points)
        
        # Lógica de detección de bostezo mejorada
        if self.current_mar > self.MOUTH_AR_THRESH and vertical_distance > 30:  # Apertura significativa
            if not self.yawning:
                self.yawning = True
                self.yawn_start_time = current_time
                cv2.polylines(frame, [lips_points], True, (0, 255, 255), 2)
            else:
                # Verificar duración del bostezo
                yawn_duration = current_time - self.yawn_start_time
                
                # Mostrar duración del bostezo actual
                cv2.putText(
                    frame,
                    f"Duracion: {yawn_duration:.1f}s",
                    (10, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                
                if yawn_duration >= self.MIN_YAWN_DURATION:
                    # Marcar como bostezo confirmado
                    cv2.polylines(frame, [lips_points], True, (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "¡BOSTEZO DETECTADO!",
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    
                    # Dibujar rectángulo rojo
                    rect = cv2.boundingRect(lips_points)
                    cv2.rectangle(
                        frame,
                        (rect[0]-10, rect[1]-10),
                        (rect[0]+rect[2]+10, rect[1]+rect[3]+10),
                        (0, 0, 255),
                        2
                    )
                    
                    # Incrementar contador solo una vez por bostezo
                    if yawn_duration >= self.MIN_YAWN_DURATION and yawn_duration < (self.MIN_YAWN_DURATION + 0.1):
                        self.total_yawns += 1
        else:
            # Resetear estado si la boca no está suficientemente abierta
            self.yawning = False
            self.yawn_start_time = None
            cv2.polylines(frame, [lips_points], True, (0, 255, 255), 1)

        # Mostrar métricas en pantalla
        cv2.putText(
            frame,
            f"MAR: {self.current_mar:.2f}",
            (frame.shape[1] - 100, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 191, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Bostezos: {self.total_yawns}",
            (10, frame.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1
        )

        return frame