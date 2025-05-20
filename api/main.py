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

COLUMNS = ['image_name','x','y','r','detection','track_id','label']
PX_SHIFT = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
SHIFT_DIRECTIONS = [(x, y) for x, y in [
    (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1)
]]

class TrackerArgs:
    def __init__(self):
        self.input = ""             # Ruta al JSON de entrada
        self.output = ""            # Ruta de salida
        self.video_path = None      # Opcional
        self.draw_tracking = False
        self.draw_circles = False
        self.radius = 10            # Valor por defecto

app = FastAPI()

@app.post("/tracker_task")
async def tracker(request: TrackerRequest):
    
    # Request del endpoint
    input_path = request.input_folder                # ORIGINALMENTE ES EL PATH DEL JSON
    output_path = request.output_folder
    radius = request.radius
    video_name = request.video_name                    
    draw_circles = request.draw_circles
    draw_tracking = request.draw_tracking
    
    video_file = os.path.join(input_path, video_name)
    json_file = os.path.join(input_path, str(video_name).replace('.mp4', '.json'))

    json_name = os.path.basename(json_file)
    
    # Convertimos la request del Endpoint a los args que requiere el tracker
    args = TrackerArgs()
    args.input = json_file
    args.output = output_path
    args.radius = radius
    if video_name is not None:
        args.video_path = os.path.join(input_path, video_name)
        args.draw_tracking = draw_tracking
        args.draw_circles = draw_circles

    # Cargamos el JSON
    detections_file = open(json_file, 'r')
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
            for px_shift in PX_SHIFT:    # Se prueban distintos niveles de shift               
                for x,y in SHIFT_DIRECTIONS:
                    x_shift = px_shift * x
                    y_shift = px_shift * y 
                    
                    # Copia del frame original
                    previous_copy = copy.deepcopy(previous)
                    
                    # Mueve el frame
                    shifted_cloud = to_cloud(previous_copy, x_shift, y_shift)     
                    
                    # Se evalua el nuevo frame respecto del original
                    icp_shifted = o3d.pipelines.registration.registration_icp(shifted_cloud, current_cloud, radius)
                    
                    best_fitness = icp_shifted.fitness
                    if fitness < best_fitness:
                        fitness = best_fitness
                        previous_copy_cloud = copy.deepcopy(shifted_cloud)
                        
                        if fitness > 0.8:   # Si supera el umbral entonces se corta el loop
                            break
                        
            correspondence_set = np.asarray(icp.correspondence_set)
            if fitness < 0.85:
                print(args.input[50:-5], f' fitness: {fitness} frame {tracker.frame}:')
                
        tracker.update_ids(correspondence_set, current, previous)
        
    tracker.write_results(args.output)
    
    return {"message": "Tracker ejecutado correctamente", "output_path": args.output, "video_path": args.video_path}
            
            
    
    
        
    