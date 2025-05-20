from celery import Celery
import pandas as pd
import numpy as np
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
import copy
import cv2
import os
import json
import multiprocessing
import queue

from utils import init_ids, update_ids, write_results, to_cloud

# Constantes a nivel de módulo
MIN_FITNESS_THRESHOLD = 0.8  # Umbral mínimo de fitness para considerar una coincidencia válida
GOOD_FITNESS_THRESHOLD = 0.85  # Umbral óptimo de fitness para detener la búsqueda temprana
COLUMNS = ['image_name','x','y','r','detection','track_id','label']  # Columnas del DataFrame de resultados
PX_SHIFTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]  # Valores de desplazamiento en píxeles a probar
SHIFT_DIRECTIONS = [(x, y) for x, y in [
    (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1)
]]  # Direcciones de desplazamiento a probar
NUM_WORKERS = 1 # Número de workers para Celery, ajustar según capacidades del sistema

# Configuración de Celery
app = Celery(
    'tracker',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/1',
    queue='tracker_queue'
)

app.conf.worker_concurrency = NUM_WORKERS

@app.task
def tracker_task(_ignore, input_folder, output_folder, radius, video_name, draw_circles, draw_tracking):
    try:
        video_path = os.path.join(input_folder, video_name)
        detections_path = video_path.replace('.mp4', '.json')
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        
        with open(detections_path, 'r') as detections_file:
            detections = json.load(detections_file)
        
        df = pd.DataFrame(columns=COLUMNS)
        vs = cv2.VideoCapture(video_path)

        try:
            if not vs.isOpened():
                raise ValueError(f"No se pudo abrir el video: {video_path}")
            
            id_ctr = 0
            previous_ids_dict = {}
            current_ids_dict = {}
            frame = 0
            
            for i in range(len(detections)-1):
                
                print(f'Frame {i} de {len(detections)-1}')
                
                if i == 0:
                    previous = detections[str(i)]
                    id_ctr, previous_ids_dict, df = init_ids(
                        vector=previous, 
                        id_ctr=id_ctr, 
                        previous_ids_dict=previous_ids_dict, 
                        draw_circ=draw_circles,
                        draw_tracking=draw_tracking, 
                        vs=vs,  
                        frame=frame, 
                        df=df,
                        video_name=os.path.basename(video_path),
                        output_path=output_folder)
                    continue
                
                frame += 1
                previous = detections[str(i-1)]
                current = detections[str(i)]
                
                result_queue = multiprocessing.Queue()  # Crear una cola para el resultado
                process = multiprocessing.Process(
                    target=calculate_frame_correspondence,
                    args=(previous, current, radius, result_queue)
                )
                process.start()
                process.join(timeout=60)
                
                if process.is_alive():
                    print("[ERROR] Timeout al calcular la correspondencia en el proceso.")
                    process.terminate()  # Forzar la terminación del proceso
                    fitness, correspondence_set = 0.0, np.array([])
                else:
                    result = result_queue.get()
                    if isinstance(result, Exception):
                        raise result
                    fitness, correspondence_set = result
                
                print("update_ids")
                
                current_ids_dict, previous_ids_dict, id_ctr, df = update_ids(
                    correspondence_set,
                    current,
                    previous,
                    current_ids_dict,
                    previous_ids_dict,
                    id_ctr,
                    draw_tracking,
                    vs,
                    video_name=video_name,
                    frame=frame,
                    df=df,
                    draw_circ=draw_circles,
                    output_path=output_folder
                )
        finally:
            vs.release()
        
        write_results(df=df, output=output_folder, name='tracker_detections.csv')
        
    except Exception as e:
        print(f"Error en tracker_task: {str(e)}")
        raise

def calculate_frame_correspondence(previous, current, radius, result_queue):
    try:
        previous_cloud = to_cloud(previous)
        current_cloud = to_cloud(current)
        
        print(f"previous_cloud: {len(previous_cloud.points)}")
        print(f"current_cloud: {len(current_cloud.points)}")

        if len(previous_cloud.points) == 0 or len(current_cloud.points) == 0:
            raise ValueError("Una de las nubes de puntos está vacía")

        icp = o3d.pipelines.registration.registration_icp(
            previous_cloud, 
            current_cloud, 
            radius
        )

        if icp.fitness >= MIN_FITNESS_THRESHOLD:
            print("AA")
            return icp.fitness, np.asarray(icp.correspondence_set)

        best_fitness = icp.fitness
        best_correspondence = np.asarray(icp.correspondence_set)

        print(f"best_fitness: {best_fitness}")

        for px_shift in PX_SHIFTS:
            
            print(f"px_shift: {px_shift}")
            
            for x_mult, y_mult in SHIFT_DIRECTIONS:
                x_shift = px_shift * x_mult
                y_shift = px_shift * y_mult
                
                print(f"Shift: {x_shift}, {y_shift}")

                shifted_cloud = to_cloud(previous, x_shift, y_shift)
                if len(shifted_cloud.points) == 0:
                    continue

                icp_shifted = o3d.pipelines.registration.registration_icp(
                    shifted_cloud, 
                    current_cloud, 
                    radius
                )
                
                print(icp_shifted)

                if icp_shifted.fitness > best_fitness:
                    best_fitness = icp_shifted.fitness
                    best_correspondence = np.asarray(icp_shifted.correspondence_set)

                    if best_fitness >= GOOD_FITNESS_THRESHOLD:
                        return best_fitness, best_correspondence
                    
        result_queue.put((best_fitness, best_correspondence))

    except Exception as e:
        print(f"[ERROR] En calculate_frame_correspondence: {e}")
        raise
