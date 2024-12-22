import config
import numpy as np
import globals
from utils import enlarge_box, compute_center, calculate_iou
def check_collision(tracker1, tracker2, i, j):
    """
    Controlla se due tracker stanno collidendo basandosi su IoU, distanza e deviazione.
    Aggiorna le variabili globali se viene rilevata una collisione.
    """

    pair = tuple(sorted((i, j)))

    # Evita rilevazioni multiple consecutive usando il cooldown
    if globals.collision_cooldown.get(pair, 0) > 0:
        globals.collision_cooldown[pair] -= 1
        return False

    # Ingrandisci le bounding box
    enlarged_box1 = enlarge_box(tracker1['box'], config.BOX_SCALE_FACTOR)
    enlarged_box2 = enlarge_box(tracker2['box'], config.BOX_SCALE_FACTOR)

    # Calcola IoU, distanza e deviazione
    iou = calculate_iou(enlarged_box1, enlarged_box2)
    center1_observed, center2_observed = compute_center(tracker1['box']), compute_center(tracker2['box'])
    distance = np.linalg.norm(np.array(center1_observed) - np.array(center2_observed))
    deviation1, deviation2 = calculate_deviation(tracker1, center1_observed), calculate_deviation(tracker2, center2_observed)

    # Stampa per debug
    log_collision_debug(i, j, center1_observed, center2_observed, deviation1, deviation2, iou, distance)

    # Controllo delle condizioni di collisione
    if is_collision(iou, distance, deviation1, deviation2):
        handle_collision(tracker1, tracker2, i, j, pair)
        return True

    globals.iou_non_zero_status[pair] = False
    return False


def calculate_deviation(tracker, observed_center):
    """
    Calcola la deviazione tra la posizione predetta dal Kalman Filter e quella osservata.
    """
    tracker['kf'].predict()
    predicted_center = tracker['kf'].x[:2]
    deviation = np.linalg.norm(predicted_center - np.array(observed_center))
    return deviation


def log_collision_debug(i, j, center1, center2, deviation1, deviation2, iou, distance):
    """
    Logga informazioni utili per il debug delle collisioni.
    """
    print(f"ID {i} - Observed: {center1}, Deviation: {deviation1}")
    print(f"ID {j} - Observed: {center2}, Deviation: {deviation2}")
    print(f"IoU: {iou}, Distance: {distance}")


def is_collision(iou, distance, deviation1, deviation2):
    """
    Determina se una collisione è avvenuta basandosi su soglie predefinite.
    """
    return (
        iou > config.THRESHOLD_IOU and
        distance < config.THRESHOLD_DISTANCE and
        (deviation1 > config.THRESHOLD_DEVIATION or deviation2 > config.THRESHOLD_DEVIATION)
    )


def handle_collision(tracker1, tracker2, i, j, pair):
    """
    Gestisce una collisione tra due tracker, aggiornando le variabili globali e applicando gli effetti.
    """

    if not globals.iou_non_zero_status.get(pair, False):
        globals.collisions += 1
        globals.collision_cooldown[pair] = config.COOLDOWN_FRAMES
        globals.explosion_cooldown[pair] = config.EXPLOSION_DURATION
        globals.iou_non_zero_status[pair] = True
        globals.collision_color_cooldown[i] = config.COLLISION_COLOR_DURATION
        globals.collision_color_cooldown[j] = config.COLLISION_COLOR_DURATION

        # Aggiorna HP dei tracker in base alle velocità relative
        update_tracker_hp(tracker1, tracker2)


def update_tracker_hp(tracker1, tracker2):
    """
    Aggiorna gli HP dei tracker in base alle loro velocità relative.
    """
    velocity1 = np.linalg.norm(tracker1['kf'].x[2:])
    velocity2 = np.linalg.norm(tracker2['kf'].x[2:])
    total_velocity = velocity1 + velocity2

    if total_velocity > 0:
        tracker1['hp'] -= int((velocity2 / total_velocity) * config.COLLISION_DAMAGE)
        tracker2['hp'] -= int((velocity1 / total_velocity) * config.COLLISION_DAMAGE)
