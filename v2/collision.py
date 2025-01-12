import numpy as np
from utils import enlarge_box, compute_center, calculate_iou
def check_collision(tracker1, tracker2, i, j,state):
    """
    Controlla se due tracker stanno collidendo basandosi su IoU, distanza e deviazione.
    Aggiorna le variabili globali se viene rilevata una collisione.
    """

    pair = tuple(sorted((i, j)))

    # Evita rilevazioni multiple consecutive usando il cooldown
    if state.collision_cooldown.get(pair, 0) > 0:
        state.collision_cooldown[pair] -= 1
        return False

    # Ingrandisci le bounding box
    enlarged_box1 = enlarge_box(tracker1['box'], state.BOX_SCALE_FACTOR, state.w, state.h)
    enlarged_box2 = enlarge_box(tracker2['box'], state.BOX_SCALE_FACTOR, state.w, state.h)

    # Calcola IoU, distanza e deviazione
    iou = calculate_iou(enlarged_box1, enlarged_box2)
    center1_observed, center2_observed = compute_center(tracker1['box']), compute_center(tracker2['box'])
    distance = np.linalg.norm(np.array(center1_observed) - np.array(center2_observed))
    deviation1, deviation2 = calculate_deviation(tracker1, center1_observed), calculate_deviation(tracker2, center2_observed)

    # Stampa per debug
    log_collision_debug(i, j, center1_observed, center2_observed, deviation1, deviation2, iou, distance)

    # Controllo delle condizioni di collisione
    if is_collision(iou, distance, deviation1, deviation2,state):
        handle_collision(tracker1, tracker2, i, j, pair,state)
        return True

    state.iou_non_zero_status[pair] = False
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


def is_collision(iou, distance, deviation1, deviation2,state):
    """
    Determina se una collisione è avvenuta basandosi su soglie predefinite.
    """
    return (
        iou > state.THRESHOLD_IOU and
        distance < state.THRESHOLD_DISTANCE and
        (deviation1 > state.THRESHOLD_DEVIATION or deviation2 > state.THRESHOLD_DEVIATION)
    )


def handle_collision(tracker1, tracker2, i, j, pair, state):
    """
    Gestisce una collisione tra due tracker, aggiornando le variabili globali e applicando gli effetti.
    """

    if not state.iou_non_zero_status.get(pair, False):
        print("COLLISION")
        state.collisions += 1
        state.collision_cooldown[pair] = state.COOLDOWN_FRAMES
        state.explosion_cooldown[pair] = state.EXPLOSION_DURATION
        state.iou_non_zero_status[pair] = True
        state.collision_color_cooldown[i] = state.COLLISION_COLOR_DURATION
        state.collision_color_cooldown[j] = state.COLLISION_COLOR_DURATION

        # Aggiorna HP dei tracker in base alle velocità relative
        update_tracker_hp(tracker1, tracker2, state)


def update_tracker_hp(tracker1, tracker2, state):
    """
    Aggiorna gli HP dei tracker in base alle loro velocità relative.
    """
    velocity1 = np.linalg.norm(tracker1['kf'].x[2:])
    velocity2 = np.linalg.norm(tracker2['kf'].x[2:])
    total_velocity = velocity1 + velocity2

    if total_velocity > 0:
        tracker1['hp'] -= int((velocity2 / total_velocity) * state.COLLISION_DAMAGE)
        tracker2['hp'] -= int((velocity1 / total_velocity) * state.COLLISION_DAMAGE)
