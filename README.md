# Detector de somnolencia

Visión artificial en tiempo real con la webcam: detecta parpadeos prolongados, ojos cerrados y bostezos a partir de los puntos faciales del rostro, y avisa en pantalla cuando aparecen señales de somnolencia.

Más proyectos míos en **[andersein.com](https://andersein.com)**.

## Cómo funciona

1. **Puntos faciales.** MediaPipe Face Mesh localiza los puntos del contorno de ojos y labios en cada fotograma.
2. **Ojos — EAR (*Eye Aspect Ratio*).** Relación entre la apertura vertical y la anchura del ojo. Si baja del umbral (`0,20`) durante varios fotogramas seguidos, se considera ojo cerrado; si se mantiene, alerta de somnolencia.
3. **Boca — MAR (*Mouth Aspect Ratio*).** Misma idea con varios puntos verticales de los labios. Una apertura por encima del umbral (`0,6`) mantenida en el tiempo cuenta como bostezo.
4. **Resultado en pantalla.** Contornos dibujados, valores EAR/MAR, contadores, FPS y avisos.

## Estructura

| Archivo | Qué hace |
|---|---|
| `dormido.py` | Punto de entrada: abre la cámara, ejecuta Face Mesh y combina ambos detectores. |
| `parpadeo.py` | Detector de parpadeos y ojos cerrados (EAR). |
| `bostezo.py` | Detector de bostezos (MAR) con criterio de duración. |

## Uso

```bash
pip install -r requirements.txt
python dormido.py
```

Pulsa `q` para salir. Necesita una webcam; los umbrales se ajustan al principio de cada detector.

## Tecnologías

Python · OpenCV · MediaPipe · NumPy · SciPy

## Licencia

MIT — ver [LICENSE](LICENSE).
