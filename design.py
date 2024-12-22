import numpy as np
import cv2
import globals


def load_png(png_path):
    return cv2.imread(png_path, cv2.IMREAD_UNCHANGED)  # Carica immagine con canale alfa

# Funzione per disegnare l'effetto collisione
def draw_collision_effect(frame, tracker1, tracker2, explosion_img):
    # Calcola il centro della collisione
    box1, box2 = tracker1['box'], tracker2['box']
    center_x = (max(box1[0], box2[0]) + min(box1[2], box2[2])) // 2
    center_y = (max(box1[1], box2[1]) + min(box1[3], box2[3])) // 2

    # Dimensione fissa dell'esplosione (ridimensiona solo se serve)
    explosion_size = (int(globals.w/1) , int(globals.h/1))
    resized_explosion = cv2.resize(explosion_img, explosion_size)

    # Posiziona l'esplosione centrata
    top_left_x = center_x - explosion_size[0] // 2
    top_left_y = center_y - explosion_size[1] // 2

    # Sovrapponi l'immagine sul frame
    overlay_image(frame, resized_explosion, (top_left_x, top_left_y))
    
# Funzione per sovrapporre un'immagine PNG
def overlay_image(img, overlay, position):
    x, y = position

    # Dimensioni dell'immagine originale e dell'overlay
    h, w = overlay.shape[:2]
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)

    # Corrispondenti dimensioni dell'overlay da sovrapporre
    overlay_crop = overlay[:y2-y1, :x2-x1]
    alpha = overlay_crop[:, :, 3] / 255.0  # Canale alfa
    alpha_inv = 1.0 - alpha

    # Sovrapposizione: considera solo i primi 3 canali (BGR)
    for c in range(3):
        img[y1:y2, x1:x2, c] = (alpha * overlay_crop[:, :, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

# Funzione per sovrapporre l'immagine della scintilla
def overlay_spark_image(frame, spark_image, position, alpha=1.0):
    """
    Sovrappone un'immagine di scintilla sulla scena con un certo grado di trasparenza.
    """
    spark_image = cv2.resize(spark_image, (int(globals.w/4) , int(globals.h/4)))

    # Separa i canali RGBA dell'immagine della scintilla
    spark_bgra = spark_image
    spark_rgb = spark_bgra[:, :, :3]  # RGB
    spark_alpha = spark_bgra[:, :, 3] / 255.0  # Canale Alpha normalizzato

    # Ottieni le dimensioni dell'immagine della scintilla
    spark_height, spark_width = spark_rgb.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    x, y = position

    # Verifica che la scintilla non esca dai limiti del frame
    if x + spark_width > frame_width:
        x = frame_width - spark_width
    if y + spark_height > frame_height:
        y = frame_height - spark_height

    # Sovrapposizione
    for c in range(3):  # Per ogni canale RGB
        frame[y:y+spark_height, x:x+spark_width, c] = \
            (1 - alpha * spark_alpha) * frame[y:y+spark_height, x:x+spark_width, c] + \
            alpha * spark_alpha * spark_rgb[:, :, c]
