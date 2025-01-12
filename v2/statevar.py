# state.py
import cv2
class GlobalState:
    def __init__(self):
        # Configurazione immagini
        self.EXPLOSION_IMAGE_PATH = ".././data/coll.png"
        self.SPARK_IMAGE_PATH = ".././data/spark.png"

        self.explosion_img = cv2.imread(self.EXPLOSION_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        self.spark_img = cv2.imread(self.SPARK_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        # Parametri principali
        self.w = None
        self.h = None
        self.N_BEYBLADE = 2
        self.BOX_SCALE_FACTOR = 1.1
        self.THRESHOLD_IOU = 0.00
        self.THRESHOLD_DISTANCE = None
        self.THRESHOLD_DEVIATION = 10
        self.COOLDOWN_FRAMES = 5
        self.MAX_HP = 100
        self.HP_DECAY = 1
        self.COLLISION_DAMAGE = 10

        # Parametri video
        self.VIDEO_PATH = ".././data/videoB.mp4"
        self.OUTPUT_PATH = "scie5.mp4"
        self.FRAME_INTERVAL = 300

        # Durate e cooldown
        self.COLLISION_COLOR_DURATION = 10
        self.EXPLOSION_DURATION = 2

        # Variabili globali
        self.trajectories = {i:[] for i in range(self.N_BEYBLADE)}
        self.lost_trackers = {}
        self.collisions = 0
        self.collision_cooldown = {}
        self.trackers = {}
        self.iou_non_zero_status = {}
        self.prev_colors = {}
        self.collision_color_cooldown = {}
        self.explosion_cooldown = {}
        self.frame_count = 0
