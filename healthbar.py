import cv2
import numpy as np

# Parametri iniziali
screen_width = 1280  # Larghezza dello schermo
screen_height = 720  # Altezza dello schermo
max_hp = 100         # Vita massima delle trottole
v1 = 10              # Valore di vita perso ad ogni collisione

# Stato iniziale della vita delle trottole
hp_beyblade1 = max_hp
hp_beyblade2 = max_hp

# Colori per la barra della vita
color_bg = (50, 50, 50)  # Grigio per lo sfondo della barra
color_text = (255, 255, 255)  # Colore del testo (bianco)

# Funzione per calcolare il colore della barra in base alla percentuale di vita
def calculate_bar_color(hp, max_hp):
    """
    Calcola il colore della barra interpolando tra verde e rosso.
    """
    ratio = hp / max_hp  # Percentuale di vita rimanente
    r = int((1 - ratio) * 255)  # Maggiore il danno, maggiore il valore di rosso
    g = int(ratio * 255)        # Minore il danno, maggiore il valore di verde
    return (0, g, r)            # Restituisce il colore in formato BGR (OpenCV usa BGR)

# Funzione per disegnare la barra della vita con il nome del giocatore
def draw_health_bar_with_label(frame, hp, max_hp, x, y, width, height, label, color_bg, color_text):
    """
    Disegna una barra della vita con una label sopra di essa, cambiando colore dinamicamente.
    """
    # Posizione del testo (label del giocatore)
    text_position = (x, y - 10)  # 10 pixel sopra la barra
    cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2, cv2.LINE_AA)

    # Calcola il colore della barra in base all'HP
    color_hp = calculate_bar_color(hp, max_hp)

    # Disegna lo sfondo della barra
    cv2.rectangle(frame, (x, y), (x + width, y + height), color_bg, -1)

    # Calcola la lunghezza della barra in base all'HP rimanente
    current_width = int((hp / max_hp) * width)

    # Disegna la barra della vita
    cv2.rectangle(frame, (x, y), (x + current_width, y + height), color_hp, -1)

    # Disegna il contorno della barra
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

# Simulazione del gioco
def simulate_game():
    global hp_beyblade1, hp_beyblade2  # Variabili globali per la vita delle trottole

    # Creazione di una finestra per visualizzare la simulazione
    cv2.namedWindow("Game Simulation")

    # Loop di simulazione
    while True:
        # Crea un frame vuoto (schermo nero)
        frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # Posizioni delle barre della vita
        bar_width = 300
        bar_height = 30
        beyblade1_bar_pos = (50, 50)  # In alto a sinistra
        beyblade2_bar_pos = (screen_width - bar_width - 50, 50)  # In alto a destra

        # Disegna le barre della vita con le label
        draw_health_bar_with_label(frame, hp_beyblade1, max_hp, beyblade1_bar_pos[0], beyblade1_bar_pos[1], 
                                   bar_width, bar_height, "Player 1", color_bg, color_text)
        draw_health_bar_with_label(frame, hp_beyblade2, max_hp, beyblade2_bar_pos[0], beyblade2_bar_pos[1], 
                                   bar_width, bar_height, "Player 2", color_bg, color_text)

        # Simula una collisione (ad esempio ogni 30 frame)
        if cv2.waitKey(100) & 0xFF == ord('c'):  # Premi 'c' per simulare una collisione
            if hp_beyblade1 > 0:
                hp_beyblade1 = max(hp_beyblade1 - v1, 0)  # Sottrai vita alla trottola 1
            if hp_beyblade2 > 0:
                hp_beyblade2 = max(hp_beyblade2 - v1, 0)  # Sottrai vita alla trottola 2

        # Mostra il frame aggiornato
        cv2.imshow("Game Simulation", frame)

        # Esci dalla simulazione premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Distruggi la finestra
    cv2.destroyAllWindows()

# Avvia la simulazione
simulate_game()
