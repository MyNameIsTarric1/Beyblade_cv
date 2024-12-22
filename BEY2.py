import torch
import cv2
import numpy as np
from ultralytics import YOLO
import random
from design import load_png, draw_collision_effect
from utils import compute_center, generate_colors, overlay_spark_image, calculate_dynamic_distance_threshold
from collision import check_collision
import config
from match import match_trackers_to_detections

# Carica immagini
explosion_img = load_png(config.EXPLOSION_IMAGE_PATH)
spark_image = load_png(config.SPARK_IMAGE_PATH)

# Carica il modello YOLO
model = YOLO('last.pt')

# Carica il video
cap = cv2.VideoCapture(config.VIDEO_PATH)

# Parametri video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Variabili globali
trajectories = {i: [] for i in range(config.N_BEYBLADE)}
lost_trackers = {}
frame_count = 0
trackers = {}
prev_colors = {}

# Genera colori distinti per il numero massimo di trottole
tracker_colors = generate_colors(config.N_BEYBLADE)
                                    
# Loop principale
first_frame = True
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if frame_count % config.FRAME_INTERVAL == 0 and len(detections) >= 2:
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

    for pair, cooldown in list(globals.explosion_cooldown.items()):
        if cooldown > 0:
            i, j = pair
            tracker1 = trackers.get(i)
            tracker2 = trackers.get(j)
            if tracker1 and tracker2:  # Verifica che i tracker esistano
                draw_collision_effect(frame, tracker1, tracker2)
            globals.explosion_cooldown[pair] -= 1  # Decrementa il cooldown
        else:
            del globals.explosion_cooldown[pair]

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
            hp_ratio = tracker['hp'] / config.MAX_HP
            current_bar_y1 = y2 - int((y2 - y1) * hp_ratio)  # Riduci dall'alto verso il basso
            
            # Disegna la barra dello sfondo (grigia)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
            
            # Disegna la barra della vita (verde) dall'alto in base all'HP rimanente
            cv2.rectangle(frame, (bar_x1, current_bar_y1), (bar_x2, bar_y2), (0, 255, 0), -1)

            # Aggiorna gli HP riducendoli gradualmente ogni secondo
            if frame_count % fps == 0:
                tracker['hp'] = max(tracker['hp'] - config.HP_DECAY, 0)

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
    cv2.putText(frame, f"COLLISIONI = {globals.collisions}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    out.write(frame)
    cv2.imshow('Tracking e Collisioni', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
    