from utils import init_kalman, compute_center, calculate_iou, calculate_average_color, color_distance
import numpy as np
import config
def calculate_matching_score(box, tracker, frame):
    """
    Calcola il punteggio di corrispondenza tra una detection e un tracker esistente.
    Il punteggio si basa su distanza, colore, e IoU.
    """
    center = compute_center(box)
    avg_color = calculate_average_color(frame, box)
    color_dist = color_distance(avg_color, tracker['avg_color'])

    distance = np.linalg.norm(np.array(center) - np.array(compute_center(tracker['box'])))
    iou = calculate_iou(box, tracker['box'])
    color_dist_initial = color_distance(avg_color, tracker['initial_color'])

    # Peso dei parametri nel punteggio
    score = (distance * 0.4) - (iou * 500) + (color_dist * 40) + (color_dist_initial * 30)
    return score, center, avg_color


def match_existing_trackers(trackers, detections, frame, used_detections):
    """
    Associa i tracker esistenti alle nuove detection, aggiornando la posizione e le caratteristiche.
    """
    new_trackers = {}
    lost_trackers = {}

    for tid, tracker in trackers.items():
        best_match = None
        best_score = float('inf')

        for i, box_conf in enumerate(detections):
            if i in used_detections:
                continue

            box = list(map(int, box_conf[:4]))
            score, center, avg_color = calculate_matching_score(box, tracker, frame)

            if score < best_score:
                best_score = score
                best_match = (i, box, center, avg_color)

        if best_match:
            i, box, center, avg_color = best_match
            used_detections.add(i)
            tracker['kf'].predict()
            tracker['kf'].update(center)
            tracker.update({'box': box, 'avg_color': avg_color})
            new_trackers[tid] = tracker
        else:
            # Se il tracker non è stato associato, considera come perso
            lost_trackers[tid] = tracker

    return new_trackers, lost_trackers


def add_new_trackers(new_trackers, lost_trackers, detections, frame, used_detections, max_trackers):
    """
    Aggiunge nuovi tracker per le detection non assegnate o riacquisisce tracker persi.
    """
    next_id = len(new_trackers)

    for i, box_conf in enumerate(detections):
        if i not in used_detections and len(new_trackers) < max_trackers:
            box = list(map(int, box_conf[:4]))
            center = compute_center(box)
            avg_color = calculate_average_color(frame, box)
            kf = init_kalman()
            kf.x[:2] = center

            # Se il tracker è stato perso, recupera gli attributi precedenti
            if next_id in lost_trackers:
                hp = lost_trackers[next_id]['hp']
                del lost_trackers[next_id]
            else:
                hp = config.MAX_HP  # Nuovo tracker con HP iniziale

            # Aggiungi il nuovo tracker
            new_trackers[next_id] = {
                'kf': kf,
                'box': box,
                'avg_color': avg_color,
                'initial_color': avg_color,  # Firma iniziale del colore
                'frozen': 0,
                'stability': 1.0,
                'hp': hp
            }
            next_id += 1

    return new_trackers


def match_trackers_to_detections(trackers, detections, frame):
    """
    Aggiorna i tracker esistenti con le nuove detection e aggiunge nuovi tracker se necessario.
    """
    used_detections = set()

    # Associazioni tra tracker esistenti e detection
    new_trackers, lost_trackers = match_existing_trackers(trackers, detections, frame, used_detections)

    # Aggiunta di nuovi tracker
    new_trackers = add_new_trackers(new_trackers, lost_trackers, detections, frame, used_detections, config.N_BEYBLADE)

    return new_trackers
