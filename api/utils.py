
import numpy as np
import open3d as o3d

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