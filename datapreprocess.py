import cv2
import os
import glob


def extract_frames(video_path, store):
    currentframe = 0
    
    # Apri il video
    vid = cv2.VideoCapture(video_path)
    
    # Controlla se il video è stato caricato correttamente
    if not vid.isOpened():
        print(f"Errore: Impossibile aprire il video {video_path}")
        return

    while True:
        # Leggi il frame
        success, frame = vid.read()

        # Verifica se il frame è stato letto correttamente
        if not success:
            print("Fine del video o errore nella lettura del frame.")
            break

        # Mostra il frame
        cv2.imshow("Output", frame)

        # Salva il frame come immagine
        frame_path = f"{store}/frame{currentframe}.jpg"
        cv2.imwrite(frame_path, frame)
        currentframe += 1

        # Esci quando si preme il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    vid.release()
    cv2.destroyAllWindows()
    return currentframe

from PIL import Image
import os

def rotate_resize(input_folder, output_folder):
	# Controlla che la cartella di output esista, altrimenti la crea
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	# Itera attraverso tutti i file nella cartella
	for filename in os.listdir(input_folder):
		if filename.lower().endswith(('.jpg')):
			try:
				# Percorso completo del file
				file_path = os.path.join(input_folder, filename)
				
				# Apri l'immagine
				img = Image.open(file_path)
				
				# Ruota l'immagine di 90 gradi
				# img_rotated = img.rotate(90, expand=True)
				
				# Ridimensiona l'immagine a 640x640
				img_resized = img.resize((640, 640))
				
				# Salva l'immagine nella cartella di output
				output_path = os.path.join(output_folder, filename)
				img_resized.save(output_path)
				
				print(f"Processata immagine: {filename}")
			except Exception as e:
				print(f"Errore con l'immagine {filename}: {e}")


# crea le bbox e il corrispondente file yolo
def process_images_and_create_yolo_annotations(frame_store, bbox_store, label_directory):
	# Crea le directory di output se non esistono
	os.makedirs(bbox_store, exist_ok=True)
	os.makedirs(label_directory, exist_ok=True)

	print(f"Elaborazione frame da: {frame_store}")
	print(f"Salvataggio bounding box in: {bbox_store}")
	print(f"Salvataggio annotazioni YOLO in: {label_directory}")

	# Ordina e carica i frame
	frames = sorted(glob.glob(f"{frame_store}/*.jpg"))
	print(f"Trovati {len(frames)} frame da processare.")

	fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

	for i, frame_path in enumerate(frames):
		# Carica l'immagine
		image = cv2.imread(frame_path)
		if image is None:
			print(f"[ERRORE] Impossibile caricare l'immagine: {frame_path}")
			continue

		# Applica il sottrattore di sfondo
		fgmask = fgbg.apply(image)
		
		# Pulizia dell'immagine binaria
		fgmask = cv2.medianBlur(fgmask, 5)

		# Trova i contorni dell'oggetto
		contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if contours:
			# Trova il contorno più grande
			largest_contour = max(contours, key=cv2.contourArea)
			x, y, w, h = cv2.boundingRect(largest_contour)

			# Disegna la bounding box sull'immagine
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

			# Crea il file YOLO direttamente qui
			height, width, _ = image.shape
			x_center = (x + w / 2) / width
			y_center = (y + h / 2) / height
			norm_width = w / width
			norm_height = h / height

			# Nome del file di annotazione
			filename = os.path.basename(frame_path).replace(".jpg", ".txt")
			label_path = os.path.join(label_directory, filename)

			# Scrivi il file YOLO
			with open(label_path, "w") as label_file:
				label_file.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

		# Salva il frame con la bounding box
		output_frame_path = os.path.join(bbox_store, f"frame_{i:05d}.jpg")
		cv2.imwrite(output_frame_path, image)

	print(f"[INFO] Completato per: {frame_store}\n")


def cleanup_labels(bbox_root, labels_root, log_file):
    """
    Rimuove i file di annotazione YOLO (file .txt) se le immagini corrispondenti nella cartella bbox sono state eliminate.
    Scrive un log con i nomi dei file cancellati.

    :param bbox_root: Percorso della directory che contiene le immagini bbox.
    :param labels_root: Percorso della directory che contiene i file di annotazione YOLO.
    :param log_file: Percorso del file di log dove scrivere i nomi dei file cancellati.
    """
    deleted_labels = []

    # Assicurati che la directory dei label esista
    if not os.path.exists(labels_root):
        print(f"[ERRORE] Directory dei label non trovata: {labels_root}")
        return

    # Scorri i file nella cartella labels
    for label_file in os.listdir(labels_root):
        if label_file.endswith(".txt"):
            # Trova il nome corrispondente nella cartella bbox
            # Estrarre il numero dal file .txt (ad esempio frame0.txt diventa 0)
            label_base = label_file.replace("frame", "").replace(".txt", "")
            image_file = f"frame_{int(label_base):05d}.jpg"  # Formato del frame (es. frame_00000.jpg)
            image_path = os.path.join(bbox_root, image_file)
            label_path = os.path.join(labels_root, label_file)

            # Se l'immagine bbox non esiste, cancella il file label
            if not os.path.exists(image_path):
                os.remove(label_path)
                deleted_labels.append(label_file)
                print(f"[INFO] File label rimosso: {label_path}")

    # Scrivi il log dei file cancellati
    with open(log_file, "a") as log:  # Usa "a" per appendere al file esistente
        for label in deleted_labels:
            log.write(f"{label}\n")
    print(f"[INFO] Log creato: {log_file}")
    print(f"[INFO] Totale file cancellati: {len(deleted_labels)}")



	

