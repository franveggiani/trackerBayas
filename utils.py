import cv2
import os
import copy
import numpy as np
import open3d as o3d

def draw_circle(img, center, radius, color=(23, 220, 75), thickness=1):
    img = cv2.circle(img, center, radius, color, thickness)

def draw_circles(img, df, video_name: str, frame: int, draw_circles: bool, output_path: str):
    data = df
    labels = data[data['image_name']==video_name+f'_{frame}.png']
    # ['image_name', 'x', 'y', 'r', 'detection', 'track_id', 'label']
    
    # Rotar frame 90 grados. Reemplazar rotated_frame por img
    rotated_frame = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    for _, label in labels.iterrows():
        image_name, x, y, r, _ , track_id, _ = list(label)
        center = (round(x), round(y),)
        radius = round(r)
        if draw_circles:
            draw_circle(rotated_frame, center, radius)
        text = f"{track_id}"
        cv2.putText(rotated_frame, text, (round(x)-5, round(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2 ) 
        output_img_path = os.path.join(output_path, video_name + '_' + str(frame) + '_track.png')              
    cv2.imwrite(output_img_path, rotated_frame)

def init_ids(video_path, vector, id_ctr:int, previous_ids_dict:dict, draw_tracking:bool = False, vs=None, video_name: str = '', frame: int = 0, df=None):
    # Retornar id_ctr para actualizar
    id_ctr = len(vector) - 1 # Contador de id
    for i, v in enumerate(vector.values()):
        # Retornar esto también para actualizar 
        previous_ids_dict[i] = i     # diccionario clave: previus detection index; valor:id
                                            # todas las detecciones anteriores tienen id
        df = add_to_bundle(v[0], v[1], v[2], i, video_name, frame, df)
    if draw_tracking:
        flag, img = vs.read()
        if not flag:
            raise FileNotFoundError(f"Error: No se pudo leer el video {video_path}")
        draw_circles(img)
        
    return id_ctr, previous_ids_dict, df
        
def add_to_bundle(x, y, radius, track_id, video_name, frame, df):
    #['image_name', 'x', 'y', 'r', 'detection', 'track_id', 'label']
    det = [f'{video_name}_{frame}.png', x, y, radius,'detecting', track_id, 'baya']
    df.loc[len(df)] = det
    
    return df
    
def update_ids(correspondence_set,
               current,
               previous,
               current_ids_dict:dict = {},
               previous_ids_dict:dict = {},
               id_ctr:int = 0,
               draw_tracking:bool = False,
               vs=None, 
               video_name: str = '',
               frame: int = 0,
               df=None
               ):
    # print(f'Frame: {self.frame}')
    current_ids_dict.clear()
    match_dict = { cv[0]:cv[1] for cv in correspondence_set}
    previous_matched = correspondence_set[:,0]
    current_matched = correspondence_set[:,1]
    for idx, _ in enumerate(previous):
        if idx in previous_matched:
            current_ids_dict[match_dict[idx]] = previous_ids_dict[idx]

    for idx, cv in enumerate(current.values()):
        if idx not in current_matched:
            id_ctr +=1
            current_ids_dict[idx] = id_ctr
            df = add_to_bundle(cv[0], cv[1], cv[2], current_ids_dict[idx], video_name=video_name, frame=frame, df=df)
        else:
            df = add_to_bundle(cv[0], cv[1], cv[2], current_ids_dict[idx])
    
    if draw_tracking:
        frame = vs.read()
        flag, img = frame
        if not flag:
            raise Exception()
        draw_circles(img)
    previous_ids_dict = copy.deepcopy(current_ids_dict)
    
    return current_ids_dict, previous_ids_dict, id_ctr, df
    
def write_results(df, output_path, name='tracker_detections.csv'):
    
    csv_path = os.path.join(output_path, name)
        
    print(csv_path)
    df.sort_values('track_id', inplace=True)
    df.to_csv(csv_path)
    
def to_cloud(frame_detections_dict, x=0, y=0):
    # print(frame_detections_dict)
    vector = [[det[0]+x, det[1]+y,  0.] for det in frame_detections_dict.values()]
    vector = np.array(vector)
    return conform_point_cloud(vector)

def conform_point_cloud(points):
    """
    create a PointCloud object from a matrix
    inputs:
        points: a mumpy matrix with shape (n, 3) (n arbitrary points and x, y, z coordinates)
    return:
        PointCloud object (open3d)
    """
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

def point_cloud_viewer(pcs):
    clouds_list = []
    for i, pc in enumerate(pcs):
        clouds_list.append({
            "name": f"{i}",
            "geometry": pc
        })
    o3d.visualization.draw(clouds_list, show_ui=True, point_size=7)