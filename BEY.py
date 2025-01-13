import torch
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import random


def calculate_dynamic_distance_threshold(detections):

    global  w , h
    
    if len(detections) == 0:
        return 100  # Valore predefinito di fallback

    widths = []
    heights = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])
        width = x2 - x1
        height = y2 - y1
        widths.append(width)
        heights.append(height)

    w = np.mean(widths)
    h = np.mean(heights)

    # Soglia dinamica basata su larghezza media della bounding box
    dynamic_threshold = 1.2 * w  # Puoi aggiustare il fattore moltiplicativo
    print(f"Threshold distanza calcolata dinamicamente: {dynamic_threshold:.2f} (Larghezza media: {w:.2f})")
    return dynamic_threshold
    
    
# Funzione per caricare un'immagine PNG
def load_png(png_path):
    return cv2.imread(png_path, cv2.IMREAD_UNCHANGED)  # Carica immagine con canale alfa

# Carica l'immagine dell'esplosione
explosion_img = load_png(".././data/coll.png")
spark_image = load_png(".././data/spark.png")  # Immagine PNG con trasparenza

# Funzione per disegnare l'effetto collisione
def draw_collision_effect(frame, tracker1, tracker2):
    # Calcola il centro della collisione
    box1, box2 = tracker1['box'], tracker2['box']
    center_x = (max(box1[0], box2[0]) + min(box1[2], box2[2])) // 2
    center_y = (max(box1[1], box2[1]) + min(box1[3], box2[3])) // 2

    # Dimensione fissa dell'esplosione (ridimensiona solo se serve)
    explosion_size = (int(w/1) , int(h/1))
    resized_explosion = cv2.resize(explosion_img, explosion_size)

    # Posiziona l'esplosione centrata
    top_left_x = center_x - explosion_size[0] // 2
    top_left_y = center_y - explosion_size[1] // 2

    # Sovrapponi l'immagine sul frame
    overlay_image(frame, resized_explosion, (top_left_x, top_left_y))

# Variabili principali
N_BEYBLADE = 2  # Numero massimo di trottole
BOX_SCALE_FACTOR = 1.1  # Fattore di scala per ingrandire le bounding box
THRESHOLD_IOU = 0.00  # Soglia IoU per rilevare collisioni
THRESHOLD_DISTANCE = None  # Soglia distanza tra box
THRESHOLD_DEVIATION = 10
COOLDOWN_FRAMES = 5  # Numero di frame di cooldown per le collisioni
MAX_HP = 100
HP_DECAY = 1
COLLISION_DAMAGE = 10

# Carica il modello YOLO
model = YOLO('.././data/last.pt')

# Carica il video
video_path = 'videoB.mp4'
cap = cv2.VideoCapture(video_path)

# Parametri video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = 'scie5.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Variabili globali
trajectories = {i: [] for i in range(N_BEYBLADE)}
lost_trackers = {}
collisions = 0
collision_cooldown = {}  # Pair collision cooldown
trackers = {}
iou_non_zero_status = {}  # Stato per IoU
prev_colors = {}
collision_color_cooldown = {}
COLLISION_COLOR_DURATION = 10  # Numero di frame per cui la box rimane rossa
explosion_cooldown = {}  # Cooldown per le esplosioni
EXPLOSION_DURATION = 2  # Numero di frame per cui l'esplosione rimane visibile
FRAME_INTERVAL = 300  # Numero di frame tra un ricalcolo e l'altro
frame_count = 0


# Funzioni principali
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 1000.0
    kf.R = np.eye(2) * 10
    kf.Q = np.eye(4) * 0.1
    kf.x = np.array([0, 0, 0, 0])
    return kf

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

def enlarge_box(box, scale_factor):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - w * (scale_factor - 1) / 2))
    y1 = max(0, int(y1 - h * (scale_factor - 1) / 2))
    x2 = min(width, int(x2 + w * (scale_factor - 1) / 2))
    y2 = min(height, int(y2 + h * (scale_factor - 1) / 2))
    return [x1, y1, x2, y2]

def calculate_color_histogram(frame, box):
    x1, y1, x2, y2 = box
    cropped = frame[y1:y2, x1:x2]
    hist = cv2.calcHist([cropped], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def calculate_average_color(frame, box):
    x1, y1, x2, y2 = box
    cropped = frame[y1:y2, x1:x2]
    avg_color = np.mean(cropped, axis=(0, 1))  # Media canali BGR
    return avg_color

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def generate_colors(num_colors):
    """Genera una lista di colori RGB casuali."""
    random.seed(42)  # Per risultati riproducibili
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]

# Genera colori distinti per il numero massimo di trottole
tracker_colors = generate_colors(N_BEYBLADE)

# Funzione per sovrapporre l'immagine della scintilla
def overlay_spark_image(frame, spark_image, position, alpha=1.0):
    """
    Sovrappone un'immagine di scintilla sulla scena con un certo grado di trasparenza.
    """
    spark_image = cv2.resize(spark_image, (int(w/4) , int(h/4)))

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
                                    
def match_trackers_to_detections(trackers, detections, frame):
    new_trackers = {}
    used_detections = set()

    for tid, tracker in trackers.items():
        best_match = None
        best_score = float('inf')

        for i, box_conf in enumerate(detections):
            if i in used_detections:
                continue

            box = list(map(int, box_conf[:4]))
            center = compute_center(box)
            avg_color = calculate_average_color(frame, box)
            color_dist = color_distance(avg_color, tracker['avg_color'])

            # Score basato su distanza, colore e IoU
            distance = np.linalg.norm(np.array(center) - np.array(compute_center(tracker['box'])))
            iou = calculate_iou(box, tracker['box'])
            color_dist_initial = color_distance(avg_color, tracker['initial_color'])
            score = (distance * 0.4) - (iou * 500) + (color_dist * 40) + (color_dist_initial * 30)

            if score < best_score:
                best_score = score
                best_match = (i, box, center, avg_color)

        if best_match:
            i, box, center, avg_color = best_match
            used_detections.add(i)
            tracker['kf'].predict()
            tracker['kf'].update(center)
            tracker['box'] = box
            tracker['avg_color'] = avg_color
            new_trackers[tid] = tracker
        else:
            # Se il tracker non è stato trovato, salvalo in lost_trackers
            lost_trackers[tid] = tracker

    # Aggiungi nuovi tracker dai lost_trackers se sono stati riacquisiti
    next_id = len(new_trackers)
    for i, box_conf in enumerate(detections):
        if i not in used_detections and len(new_trackers) < N_BEYBLADE:
            box = list(map(int, box_conf[:4]))
            center = compute_center(box)
            avg_color = calculate_average_color(frame, box)
            kf = init_kalman()
            kf.x[:2] = center

            # Se il tracker è stato perso e poi riacquisito, recupera l'HP
            if next_id in lost_trackers:
                hp = lost_trackers[next_id]['hp']  # Ripristina HP precedente
                del lost_trackers[next_id]  # Rimuovi dai lost_trackers
            else:
                hp = MAX_HP  # Nuovo tracker, quindi inizia da MAX_HP

            new_trackers[next_id] = {
                'kf': kf,
                'box': box,
                'avg_color': avg_color,
                'initial_color': avg_color,  # Firma iniziale del colore
                'frozen': 0,
                'stability': 1.0,
                'hp': hp  # Usa HP ripristinato o inizializza a MAX_HP
            }
            next_id += 1

    return new_trackers
    
def check_collision(tracker1, tracker2, i, j):
    global collisions
    pair = tuple(sorted((i, j)))

    # Cooldown: evita rilevazioni multiple in frame consecutivi
    if collision_cooldown.get(pair, 0) > 0:
        collision_cooldown[pair] -= 1
        return False

    # Ingrandisci le bounding box per rilevare sfioramenti
    enlarged_box1 = enlarge_box(tracker1['box'], BOX_SCALE_FACTOR)
    enlarged_box2 = enlarge_box(tracker2['box'], BOX_SCALE_FACTOR)

    # Calcola IoU e distanza
    iou = calculate_iou(enlarged_box1, enlarged_box2)
    center1_observed = compute_center(tracker1['box'])
    center2_observed = compute_center(tracker2['box'])
    distance = np.linalg.norm(np.array(center1_observed) - np.array(center2_observed))

    # Calcola la posizione predetta del Kalman Filter
    tracker1['kf'].predict()
    tracker2['kf'].predict()
    predicted_center1 = tracker1['kf'].x[:2]
    predicted_center2 = tracker2['kf'].x[:2]

    # Calcola la deviazione come distanza tra predizione e osservazione
    deviation1 = np.linalg.norm(predicted_center1 - np.array(center1_observed))
    deviation2 = np.linalg.norm(predicted_center2 - np.array(center2_observed))

    # Calcola la velocità delle trottole
    velocity1 = np.linalg.norm(tracker1['kf'].x[2:])
    velocity2 = np.linalg.norm(tracker2['kf'].x[2:])

    # Stampa per debug
    print(f"ID {i} - Predicted: {predicted_center1}, Observed: {center1_observed}, Deviation: {deviation1}")
    print(f"ID {j} - Predicted: {predicted_center2}, Observed: {center2_observed}, Deviation: {deviation2}")
    print(f"IoU: {iou}, Distance: {distance}")

    # Collisione se IoU, distanza o deviazione superano le soglie
    if iou > THRESHOLD_IOU and distance < THRESHOLD_DISTANCE and (deviation1 > THRESHOLD_DEVIATION or deviation2 > THRESHOLD_DEVIATION):
        if not iou_non_zero_status.get(pair, False):
            collisions += 1
            collision_cooldown[pair] = COOLDOWN_FRAMES
            explosion_cooldown[pair] = EXPLOSION_DURATION  # Imposta cooldown per l'esplosione
            iou_non_zero_status[pair] = True
            collision_color_cooldown[i] = COLLISION_COLOR_DURATION  # Imposta cooldown per il colore
            collision_color_cooldown[j] = COLLISION_COLOR_DURATION

            # Aggiorna gli HP in base alle velocità relative
            total_velocity = velocity1 + velocity2
            if total_velocity > 0:
                tracker1['hp'] -= int((velocity2 / total_velocity) * COLLISION_DAMAGE)
                tracker2['hp'] -= int((velocity1 / total_velocity) * COLLISION_DAMAGE)

            print(f"Collision detected between {i} and {j}")
            return True
    else:
        iou_non_zero_status[pair] = False

    return False

# Loop principale
first_frame = True
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if frame_count % FRAME_INTERVAL == 0 and len(detections) >= 2:
        THRESHOLD_DISTANCE = calculate_dynamic_distance_threshold(detections)

    # Incrementa il contatore dei frame
    frame_count += 1

    # Usa una soglia di fallback se non è stata calcolata
    if THRESHOLD_DISTANCE is None:
        THRESHOLD_DISTANCE = 100  # Valore di fallback

    # Aggiorna i tracker e rileva collisioni
    trackers = match_trackers_to_detections(trackers, detections, frame)

    for i, tracker1 in trackers.items():
        for j, tracker2 in trackers.items():
            if i < j and check_collision(tracker1, tracker2, i, j):
                for t in [tracker1, tracker2]:
                    x1, y1, x2, y2 = t['box']
                    draw_collision_effect(frame, tracker1, tracker2)

    for pair, cooldown in list(explosion_cooldown.items()):
        if cooldown > 0:
            i, j = pair
            tracker1 = trackers.get(i)
            tracker2 = trackers.get(j)
            if tracker1 and tracker2:  # Verifica che i tracker esistano
                draw_collision_effect(frame, tracker1, tracker2)
            explosion_cooldown[pair] -= 1  # Decrementa il cooldown
        else:
            del explosion_cooldown[pair]

    for i, tracker in trackers.items():
        if tracker.get('visible', True):  # Usa get per evitare errori
            # Disegna la box
            x1, y1, x2, y2 = tracker['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), tracker_colors[i], 2)
            cv2.putText(frame, f"BEY {i+1} - HP: {tracker['hp']}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracker_colors[i], 2)

            # Posizioni del rettangolo della barra della vita
            bar_x1 = x2 + 5  # Posiziona la barra subito a destra della box
            bar_x2 = bar_x1 + 10  # Larghezza della barra
            bar_y1 = y1  # Parte superiore della box
            bar_y2 = y2  # Parte inferiore della box
            
            # Calcola la posizione della barra rimanente in base alla vita
            hp_ratio = tracker['hp'] / MAX_HP
            current_bar_y1 = y2 - int((y2 - y1) * hp_ratio)  # Riduci dall'alto verso il basso
            
            # Disegna la barra dello sfondo (grigia)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
            
            # Disegna la barra della vita (verde) dall'alto in base all'HP rimanente
            cv2.rectangle(frame, (bar_x1, current_bar_y1), (bar_x2, bar_y2), (0, 255, 0), -1)

            # Aggiorna gli HP riducendoli gradualmente ogni secondo
            if frame_count % fps == 0:
                tracker['hp'] = max(tracker['hp'] - HP_DECAY, 0)

            # Calcola il centro della bounding box
            center = compute_center(tracker['box'])
            trajectories[i].append(center)
            if len(trajectories[i]) > 15:  # Mantieni solo le ultime posizioni
                trajectories[i].pop(0)

            # Crea la scia continua usando le immagini di scintilla
            for j in range(len(trajectories[i]) - 1):
                pt1 = tuple(map(int, trajectories[i][j]))
                pt2 = tuple(map(int, trajectories[i][j+1]))

                # Aggiungi una scintilla ad ogni posizione lungo il segmento
                for _ in range(3):  # Numero di scintille per segmento
                    # Posizione casuale lungo il segmento
                    rand_x = random.randint(min(pt1[0], pt2[0]), max(pt1[0], pt2[0]))
                    rand_y = random.randint(min(pt1[1], pt2[1]), max(pt1[1], pt2[1]))

                    # Verifica se la scintilla è fuori dalla bounding box
                    if x1 < rand_x < x2 and y1 < rand_y < y2:
                        continue  # Se dentro la bounding box, salta questa scintilla

                    # Applica trasparenza alla scintilla
                    alpha = random.uniform(0.5, 1)  # Trasparenza
                    overlay_spark_image(frame, spark_image, (rand_x, rand_y), alpha)

    # Mostra il conteggio delle collisioni
    cv2.putText(frame, f"COLLISIONI = {collisions}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    out.write(frame)
    cv2.imshow('Tracking e Collisioni', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
    