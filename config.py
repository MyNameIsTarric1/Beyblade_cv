# Configurazione immagini
EXPLOSION_IMAGE_PATH = "580b585b2edbce24c47b26da-147539564.png"
SPARK_IMAGE_PATH = "pngtree-splash-spark-line-light-effect-png-image_6300255.png"

# Parametri principali
N_BEYBLADE = 2  # Numero massimo di trottole
BOX_SCALE_FACTOR = 1.1  # Fattore di scala per ingrandire le bounding box
THRESHOLD_IOU = 0.00  # Soglia IoU per rilevare collisioni
THRESHOLD_DISTANCE = None  # Soglia distanza tra box
THRESHOLD_DEVIATION = 10
COOLDOWN_FRAMES = 5  # Numero di frame di cooldown per le collisioni
MAX_HP = 100
HP_DECAY = 1
COLLISION_DAMAGE = 10

# Parametri video
VIDEO_PATH = "videoB.mp4"
OUTPUT_PATH = "scie5.mp4"
FRAME_INTERVAL = 300  # Numero di frame tra un ricalcolo e l'altro

# Durate e cooldown
COLLISION_COLOR_DURATION = 10  # Numero di frame per cui la box rimane rossa
EXPLOSION_DURATION = 2  # Numero di frame per cui l'esplosione rimane visibile
