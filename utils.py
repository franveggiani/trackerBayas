import cv2
import os
import copy
import numpy as np
import open3d as o3d


def draw_circle(img, center, radius, color=(23, 220, 75), thickness=1):
    """Draw a circle on an image.
    
    Args:
        img (numpy.ndarray): Input image where the circle will be drawn.
        center (tuple): (x, y) coordinates of the circle center.
        radius (int): Radius of the circle in pixels.
        color (tuple, optional): BGR color tuple. Defaults to (23, 220, 75).
        thickness (int, optional): Thickness of the circle line. Defaults to 1.
        
    Returns:
        numpy.ndarray: Image with the drawn circle.
    """
    img = cv2.circle(img, center, radius, color, thickness)
    return img


def draw_circles(img, df, video_name: str, frame: int, draw_circles: bool, output_path: str):
    """Draw circles and tracking IDs on an image frame based on detection data.
    
    Args:
        img (numpy.ndarray): Input image frame.
        df (pandas.DataFrame): DataFrame containing detection information.
        video_name (str): Name of the video being processed.
        frame (int): Current frame number.
        draw_circles (bool): Whether to draw circles around detections.
        output_path (str): Path to save the output image.
    """
    data = df
    labels = data[data['image_name'] == video_name + f'_{frame}.png']
    
    # Rotate frame 90 degrees clockwise
    rotated_frame = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    for _, label in labels.iterrows():
        image_name, x, y, r, _, track_id, _ = list(label)
        center = (round(x), round(y))
        radius = round(r)
        if draw_circles:
            draw_circle(rotated_frame, center, radius)
        text = f"{track_id}"
        cv2.putText(rotated_frame, text, (round(x)-5, round(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
        output_img_path = os.path.join(output_path, video_name + '_' + str(frame) + '_track.png')
    cv2.imwrite(output_img_path, rotated_frame)


def init_ids(vector, id_ctr: int, previous_ids_dict: dict, draw_circ: bool = False, 
            draw_tracking: bool = False, vs=None, video_name: str = '', 
            frame: int = 0, df=None, output_path: str = ''):
    """Initialize tracking IDs for new detections.
    
    Args:
        vector (dict): Dictionary of current detections.
        id_ctr (int): Current ID counter.
        previous_ids_dict (dict): Dictionary mapping previous detection indices to IDs.
        draw_circ (bool, optional): Whether to draw circles. Defaults to False.
        draw_tracking (bool, optional): Whether to draw tracking information. Defaults to False.
        vs (cv2.VideoCapture, optional): Video capture object. Defaults to None.
        video_name (str, optional): Name of the video. Defaults to ''.
        frame (int, optional): Current frame number. Defaults to 0.
        df (pandas.DataFrame, optional): DataFrame to store detection info. Defaults to None.
        output_path (str, optional): Path to save output images. Defaults to ''.
        
    Returns:
        tuple: Updated (id_ctr, previous_ids_dict, df)
    """
    id_ctr = len(vector) - 1
    for i, v in enumerate(vector.values()):
        previous_ids_dict[i] = i
        df = add_to_bundle(v[0], v[1], v[2], i, video_name, frame, df)
    
    if draw_tracking:
        flag, img = vs.read()
        if not flag:
            raise FileNotFoundError(f"Error: No se pudo leer el video {video_name}")
        draw_circles(img, df=df, video_name=video_name, frame=frame, 
                    draw_circles=draw_circ, output_path=output_path)
        
    return id_ctr, previous_ids_dict, df


def add_to_bundle(x, y, radius, track_id, video_name, frame, df):
    """Add a detection to the tracking bundle DataFrame.
    
    Args:
        x (float): x-coordinate of detection.
        y (float): y-coordinate of detection.
        radius (float): Radius of detection.
        track_id (int): Tracking ID for the detection.
        video_name (str): Name of the video.
        frame (int): Current frame number.
        df (pandas.DataFrame): DataFrame to store detection info.
        
    Returns:
        pandas.DataFrame: Updated DataFrame with new detection.
    """
    det = [f'{video_name}_{frame}.png', x, y, radius, 'detecting', track_id, 'baya']
    df.loc[len(df)] = det
    return df


def update_ids(correspondence_set, current, previous, current_ids_dict: dict = {},
              previous_ids_dict: dict = {}, id_ctr: int = 0, draw_tracking: bool = False,
              vs=None, video_name: str = '', frame: int = 0, df=None,
              draw_circ: bool = False, output_path: str = ''):
    """Update tracking IDs based on current and previous detections.
    
    Args:
        correspondence_set (numpy.ndarray): Array of matched indices between current and previous detections.
        current (dict): Current frame detections.
        previous (dict): Previous frame detections.
        current_ids_dict (dict, optional): Dictionary of current IDs. Defaults to {}.
        previous_ids_dict (dict, optional): Dictionary of previous IDs. Defaults to {}.
        id_ctr (int, optional): Current ID counter. Defaults to 0.
        draw_tracking (bool, optional): Whether to draw tracking info. Defaults to False.
        vs (cv2.VideoCapture, optional): Video capture object. Defaults to None.
        video_name (str, optional): Video name. Defaults to ''.
        frame (int, optional): Current frame number. Defaults to 0.
        df (pandas.DataFrame, optional): DataFrame for detection info. Defaults to None.
        draw_circ (bool, optional): Whether to draw circles. Defaults to False.
        output_path (str, optional): Output path for images. Defaults to ''.
        
    Returns:
        tuple: Updated (current_ids_dict, previous_ids_dict, id_ctr, df)
    """
    current_ids_dict.clear()
    match_dict = {cv[0]: cv[1] for cv in correspondence_set}
    previous_matched = correspondence_set[:, 0]
    current_matched = correspondence_set[:, 1]
    
    for idx, _ in enumerate(previous):
        if idx in previous_matched:
            current_ids_dict[match_dict[idx]] = previous_ids_dict[idx]

    for idx, cv in enumerate(current.values()):
        if idx not in current_matched:
            id_ctr += 1
            current_ids_dict[idx] = id_ctr
            df = add_to_bundle(cv[0], cv[1], cv[2], current_ids_dict[idx], 
                             video_name=video_name, frame=frame, df=df)
        else:
            df = add_to_bundle(cv[0], cv[1], cv[2], current_ids_dict[idx], 
                             video_name=video_name, frame=frame, df=df)
    
    if draw_tracking:
        flag, img = vs.read()
        if not flag:
            raise Exception("Failed to read video frame")
        draw_circles(img, df=df, video_name=video_name, frame=frame, 
                    draw_circles=draw_circ, output_path=output_path)
    
    previous_ids_dict = copy.deepcopy(current_ids_dict)
    return current_ids_dict, previous_ids_dict, id_ctr, df


def write_results(df, output_path, name='tracker_detections.csv'):
    """Write tracking results to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tracking results.
        output_path (str): Path to save the output file.
        name (str, optional): Output filename. Defaults to 'tracker_detections.csv'.
    """
    csv_path = os.path.join(output_path, name)
    df.sort_values('track_id', inplace=True)
    df.to_csv(csv_path)


def to_cloud(frame_detections_dict, x=0, y=0):
    """Convert detections to a point cloud.
    
    Args:
        frame_detections_dict (dict): Dictionary of detections.
        x (int, optional): x-offset. Defaults to 0.
        y (int, optional): y-offset. Defaults to 0.
        
    Returns:
        open3d.geometry.PointCloud: Point cloud of detections.
    """
    vector = [[det[0]+x, det[1]+y, 0.] for det in frame_detections_dict.values()]
    vector = np.array(vector)
    return conform_point_cloud(vector)


def conform_point_cloud(points):
    """Create a PointCloud object from a matrix of points.
    
    Args:
        points (numpy.ndarray): Array of shape (n, 3) containing n points with x,y,z coordinates.
        
    Returns:
        open3d.geometry.PointCloud: Point cloud object.
    """
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def point_cloud_viewer(pcs):
    """Visualize multiple point clouds.
    
    Args:
        pcs (list): List of point cloud objects to visualize.
    """
    clouds_list = []
    for i, pc in enumerate(pcs):
        clouds_list.append({
            "name": f"{i}",
            "geometry": pc
        })
    o3d.visualization.draw(clouds_list, show_ui=True, point_size=7)