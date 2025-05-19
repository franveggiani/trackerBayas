from celery import Celery
import pandas as pd
import numpy as np
import open3d as o3d
import copy
import cv2
import os
import json

from utils import init_ids, update_ids, write_results, to_cloud

# Constantes a nivel de módulo
MIN_FITNESS_THRESHOLD = 0.8  # Umbral mínimo de fitness para considerar una coincidencia válida
GOOD_FITNESS_THRESHOLD = 0.85  # Umbral óptimo de fitness para detener la búsqueda temprana
COLUMNS = ['x', 'y', 'z', 'track_id', 'video_name', 'frame']  # Columnas del DataFrame de resultados
PX_SHIFTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]  # Valores de desplazamiento en píxeles a probar
SHIFT_DIRECTIONS = [(x, y) for x, y in [
    (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1)
]]  # Direcciones de desplazamiento a probar
NUM_WORKERS = 4  # Número de workers para Celery, ajustar según capacidades del sistema

# Configuración de Celery
app = Celery(
    'qrDetector',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/1',
    worker_concurrency=NUM_WORKERS,
    queue='qr_detector_queue'
)

@app.task
def tracker_task(input_path, output_path, radius, video_name, draw_circles, draw_tracking):
    """
    Tarea principal de seguimiento (tracking) de objetos en un video.
    
    Args:
        input_path (str): Ruta del directorio de entrada que contiene el video y las detecciones.
        output_path (str): Ruta del directorio de salida para guardar los resultados.
        radius (float): Radio de búsqueda para el algoritmo ICP.
        video_name (str): Nombre del archivo de video (con extensión .mp4).
        draw_circles (bool): Bandera para indicar si se deben dibujar círculos en las detecciones.
        draw_tracking (bool): Bandera para indicar si se debe visualizar el seguimiento.
    
    Returns:
        dict: Diccionario con mensaje de estado y rutas de los archivos procesados.
    
    Raises:
        ValueError: Si no se puede abrir el archivo de video.
        Exception: Para cualquier otro error durante el procesamiento.
    """
    try:
        video_path = os.path.join(input_path, video_name)
        detections_path = video_path.replace('.mp4', '.json')
        
        # Usar administrador de contexto para manejo de archivos
        with open(detections_path, 'r') as detections_file:
            detections = json.load(detections_file)
        
        df = pd.DataFrame(columns=COLUMNS)
        
        # Usar administrador de contexto para captura de video
        with cv2.VideoCapture(video_path) as vs:
            if not vs.isOpened():
                raise ValueError(f"No se pudo abrir el video: {video_path}")
            
            id_ctr = 0
            previous_ids_dict = {}
            current_ids_dict = {}
            frame = 0
            
            for i in range(len(detections)-1):
                if i == 0:
                    previous = detections[str(i)]
                    id_ctr, previous_ids_dict, df = init_ids(
                        vector=previous, 
                        id_ctr=id_ctr, 
                        previous_ids_dict=previous_ids_dict, 
                        draw_tracking=draw_tracking, 
                        vs=vs, 
                        video_name=os.path.basename(video_path), 
                        frame=frame, 
                        df=df)
                    continue
                
                frame += 1
                previous = detections[str(i-1)]
                current = detections[str(i)]
                
                fitness, correspondence_set = calculate_frame_correspondence(
                    previous, 
                    current, 
                    radius
                )
                
                if fitness < GOOD_FITNESS_THRESHOLD:
                    print(f'{detections_path[50:-5]} fitness: {fitness} frame {frame}:')
                
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
                    df=df
                )
        
        write_results(df=df, output=output_path, name='tracker_detections.csv')
        
        return {
            "message": "Tracker ejecutado correctamente", 
            "output_path": video_path, 
            "video_path": video_path
        }
        
    except Exception as e:
        # En producción, usar un sistema de logging adecuado
        print(f"Error en tracker_task: {str(e)}")
        raise  # Relanzar para que Celery maneje el error

def calculate_frame_correspondence(previous, current, radius):
    """
    Calcula la correspondencia entre frames consecutivos usando ICP.
    
    Args:
        previous (dict): Diccionario con las detecciones del frame anterior.
        current (dict): Diccionario con las detecciones del frame actual.
        radius (float): Radio de búsqueda para el algoritmo ICP.
    
    Returns:
        tuple: (fitness_score, correspondence_set) donde:
            - fitness_score: Puntuación de coincidencia entre 0 y 1
            - correspondence_set: Conjunto de correspondencias entre puntos
    
    Notas:
        Si el fitness inicial es bajo, prueba diferentes desplazamientos para mejorar la coincidencia.
    """
    previous_cloud = to_cloud(previous)
    current_cloud = to_cloud(current)
    
    icp = o3d.pipelines.registration.registration_icp(
        previous_cloud, 
        current_cloud, 
        radius
    )
    
    if icp.fitness >= MIN_FITNESS_THRESHOLD:
        return icp.fitness, np.asarray(icp.correspondence_set)
    
    # Probar diferentes desplazamientos si el fitness es bajo
    best_fitness = icp.fitness
    best_correspondence = np.asarray(icp.correspondence_set)
    
    for px_shift in PX_SHIFTS:
        for x_mult, y_mult in SHIFT_DIRECTIONS:
            x_shift = px_shift * x_mult
            y_shift = px_shift * y_mult
            
            shifted_cloud = to_cloud(previous, x_shift, y_shift)
            icp_shifted = o3d.pipelines.registration.registration_icp(
                shifted_cloud, 
                current_cloud, 
                radius
            )
            
            if icp_shifted.fitness > best_fitness:
                best_fitness = icp_shifted.fitness
                best_correspondence = np.asarray(icp_shifted.correspondence_set)
                
                if best_fitness >= GOOD_FITNESS_THRESHOLD:
                    return best_fitness, best_correspondence
    
    return best_fitness, best_correspondence