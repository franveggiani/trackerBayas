from fastapi import FastAPI
import os
import json
import open3d as o3d
import numpy as np
import copy 
from .schemas import TrackerRequest
from .tracker import Tracker
from .utils import to_cloud

# RECORDAR: Agarrar el nombre cortando los caracteres que van después de .
# HACER QUE CUANDO VIDEO_PATH SEA NONE, NO SE PERMITA PONER NI DRAW_TRACKING NI DRAW_CIRCLES

COLUMNS = ['image_name','x','y','r','detection','track_id','label']

class TrackerArgs:
    def __init__(self):
        self.input = ""             # Ruta al JSON de entrada
        self.output = ""            # Ruta de salida
        self.video_path = None      # Opcional
        self.draw_tracking = False
        self.draw_circles = False
        self.radius = 10            # Valor por defecto

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hola, ¿cómo estamos?"}

@app.get("/tracker")
async def tracker(req: TrackerRequest):
    
    # Request del endpoint
    input_path = req.input_path                    # ORIGINALMENTE ES EL PATH DEL JSON
    output_path = req.output_path
    radius = req.radius
    video_name = req.video_name                    
    draw_circles = req.draw_circles
    draw_tracking = req.draw_tracking
    id_racimo = req.id_racimo

    if id_racimo is not None:
        json_name = f"{id_racimo}_{video_name}.json"
    else: 
        json_name = f"{video_name}.json"
        
    json_path = os.path.join(input_path, json_name)
    
    # Convertimos la request del Endpoint a los args que requiere el tracker
    args = TrackerArgs()
    args.input = json_path
    args.output = output_path
    args.radius = radius
    if video_name is not None:
        args.video_path = os.path.join(input_path, video_name + '.mp4')
        args.draw_tracking = draw_tracking
        args.draw_circles = draw_circles

    # Cargamos el JSON
    detections_file = open(json_path, 'r')
    detections = json.load(detections_file)
    detections_file.close()
    
    tracker = Tracker(COLUMNS, video_name, args)
    
    for i in range(len(detections)-1):
        if i==0:
            previous = detections[str(i)]
            tracker.init_ids(previous)
            continue

        tracker.frame += 1
        previous = detections[str(i-1)]
        current = detections[str(i)]
        
        previous_cloud = to_cloud(previous)
        current_cloud = to_cloud(current)
        
        icp = o3d.pipelines.registration.registration_icp(previous_cloud, current_cloud, radius)
        correspondence_set = np.asarray(icp.correspondence_set)
        
        # Movimiento de frames para ajuste
        fitness = icp.fitness
        if fitness < 0.8:
        
            for px_shift in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:    # Se prueban distintos niveles de shift               
                for x,y in [(px_shift, 0), (px_shift, px_shift), (0, px_shift), (-px_shift, 0), (-px_shift, -px_shift), (0, -px_shift), (px_shift, -px_shift), (-px_shift, px_shift)]:
                    # Copia del frame original
                    previous_copy = copy.deepcopy(previous)
                    
                    # Mueve el frame
                    previous_copy_cloud_aux = to_cloud(previous_copy, x, y)     
                    
                    # Se evalua el nuevo frame respecto del original
                    icp2 = o3d.pipelines.registration.registration_icp(previous_copy_cloud_aux, current_cloud, radius)
                    new_fitness = icp2.fitness
                    if fitness < new_fitness:
                        fitness = new_fitness
                        previous_copy_cloud = copy.deepcopy(previous_copy_cloud_aux)
                        if fitness > 0.8:   # Si supera el umbral entonces se corta el loop
                            break
                        
            correspondence_set = np.asarray(icp.correspondence_set)
            if fitness < 0.85:
                print(args.input[50:-5], f' fitness: {fitness} frame {tracker.frame}:')
                
        tracker.update_ids(correspondence_set, current, previous)
        
    tracker.write_results(args.output)
    
        
            
            
    
    
        
    