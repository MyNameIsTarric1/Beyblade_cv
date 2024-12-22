import torch
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import random

# Funzione per calcolare la soglia dinamica della distanza tra bounding box
def calculate_dynamic_distance_threshold(detections):
    """Calcola una soglia dinamica basata sulle dimensioni medie delle bounding box."""
    global w, h

    if len(detections) == 0:
        return 100  # Valore predefinito di fallback

    widths, heights = [], []

    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])
        widths.append(x2 - x1)
        heights.append(y2 - y1)

    w, h = np.mean(widths), np.mean(heights)
    dynamic_threshold = 1.2 * w  # Fattore moltiplicativo per la soglia dinamica
    print(f"Soglia dinamica calcolata: {dynamic_threshold:.2f} (Larghezza media: {w:.2f})")
    return dynamic_threshold

# Funzione per caricare un'immagine PNG con trasparenza
def load_png(png_path):
    return cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

# Caricamento delle immagini PNG
explosion_img = load_png("580b585b2edbce24c47b26da-147539564.png")
spark_image = load_png("pngtree-splash-spark-line-light-effect-png-image_6300255.png")

# Funzione per sovrapporre immagini PNG trasparenti

def overlay_image(img, overlay, position):
    """Sovrappone un'immagine con trasparenza su un frame alla posizione specificata."""
    x, y = position
    h, w = overlay.shape[:2]
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)

    overlay_crop = overlay[:y2-y1, :x2-x1]
    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(3):
        img[y1:y2, x1:x2, c] = (alpha * overlay_crop[:, :, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

# Funzione per disegnare l'effetto collisione
def draw_collision_effect(frame, tracker1, tracker2):
    """Disegna un effetto esplosione sulla collisione tra due tracker."""
    box1, box2 = tracker1['box'], tracker2['box']
    center_x = (max(box1[0], box2[0]) + min(box1[2], box2[2])) // 2
    center_y = (max(box1[1], box2[1]) + min(box1[3], box2[3])) // 2

    explosion_size = (int(w / 1), int(h / 1))
    resized_explosion = cv2.resize(explosion_img, explosion_size)

    top_left_x = center_x - explosion_size[0] // 2
    top_left_y = center_y - explosion_size[1] // 2

    overlay_image(frame, resized_explosion, (top_left_x, top_left_y))

# Funzione per inizializzare un filtro di Kalman
def init_kalman():
    """Inizializza un filtro di Kalman con parametri standard per tracking."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 1000.0
    kf.R = np.eye(2) * 10
    kf.Q = np.eye(4) * 0.1
    kf.x = np.array([0, 0, 0, 0])
    return kf

# Funzioni di supporto per calcoli geometrici e colori
def compute_center(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

# Funzione per generare colori casuali per i tracker
def generate_colors(num_colors):
    random.seed(42)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]

# Parametri principali
N_BEYBLADE = 2
BOX_SCALE_FACTOR = 1.1
THRESHOLD_IOU = 0.00
THRESHOLD_DEVIATION = 10
COOLDOWN_FRAMES = 5
MAX_HP = 100
HP_DECAY = 1
COLLISION_DAMAGE = 10

# Caricamento modello YOLO
model = YOLO('last.pt')

# Caricamento video
video_path = 'videoB.mp4'
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('scie5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Variabili globali
trackers, lost_trackers, trajectories = {}, {}, {i: [] for i in range(N_BEYBLADE)}
iou_non_zero_status, collision_color_cooldown = {}, {}
explosion_cooldown, collision_cooldown = {}, {}
collisions, frame_count, THRESHOLD_DISTANCE = 0, 0, None
tracker_colors = generate_colors(N_BEYBLADE)

# Loop principale
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if frame_count % 300 == 0 and len(detections) >= 2:
        THRESHOLD_DISTANCE = calculate_dynamic_distance_threshold(detections)

    # Se la soglia non è calcolata, usa un valore di fallback
    if THRESHOLD_DISTANCE is None:
        THRESHOLD_DISTANCE = 100

    # Aggiorna tracker e rileva collisioni
    trackers = match_trackers_to_detections(trackers, detections, frame)

    for i, tracker1 in trackers.items():
        for j, tracker2 in trackers.items():
            if i < j and check_collision(tracker1, tracker2, i, j):
                draw_collision_effect(frame, tracker1, tracker2)

    # Disegna effetti e informazioni sui tracker
    for i, tracker in trackers.items():
        if tracker['hp'] > 0:
            x1, y1, x2, y2 = tracker['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), tracker_colors[i], 2)
            cv2.putText(frame, f"BEY {i+1} - HP: {tracker['hp']}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracker_colors[i], 2)

    # Mostra il frame
    out.write(frame)
    cv2.imshow('Tracking e Collisioni', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
