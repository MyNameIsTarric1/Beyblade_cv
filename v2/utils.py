from filterpy.kalman import KalmanFilter
import cv2
import numpy as np
import random
# Funzioni principali

def calculate_dynamic_distance_threshold(detections, state):

    
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

    state.w = np.mean(widths)
    state.h = np.mean(heights)

    # Soglia dinamica basata su larghezza media della bounding box
    dynamic_threshold = 1.2 * state.w  # Puoi aggiustare il fattore moltiplicativo
    print(f"Threshold distanza calcolata dinamicamente: {dynamic_threshold:.2f} (Larghezza media: {state.w:.2f})")
    return dynamic_threshold

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

def enlarge_box(box, scale_factor,width,height):
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

def match_trackers_to_detections(trackers, detections, frame, state):
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
            state.lost_trackers[tid] = tracker

    # Aggiungi nuovi tracker dai lost_trackers se sono stati riacquisiti
    next_id = len(new_trackers)
    for i, box_conf in enumerate(detections):
        if i not in used_detections and len(new_trackers) < state.N_BEYBLADE:
            box = list(map(int, box_conf[:4]))
            center = compute_center(box)
            avg_color = calculate_average_color(frame, box)
            kf = init_kalman()
            kf.x[:2] = center

            # Se il tracker è stato perso e poi riacquisito, recupera l'HP
            if next_id in state.lost_trackers:
                hp = state.lost_trackers[next_id]['hp']  # Ripristina HP precedente
                del state.lost_trackers[next_id]  # Rimuovi dai lost_trackers
            else:
                hp = state.MAX_HP  # Nuovo tracker, quindi inizia da MAX_HP

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
