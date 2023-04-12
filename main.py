import argparse
import json
import os
import numpy as np
import open3d as o3d
from tracker import Tracker
import pandas as pd
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='el path al json')
parser.add_argument('--output', type=str, help='path de salida')
parser.add_argument('--video_path', type=str, help='path al video, por si queremos guardar imágenes', default=None)

# parser.add_argument('--draw_detections', help='dibuja detecciones', default=False, action='store_true')
parser.add_argument('--draw_tracking', help='dibuja trackeo', default=False, action='store_true')
parser.add_argument('--draw_circles', help='dibuja trackeo', default=False, action='store_true')
parser.add_argument('--radius', type=int, help='radio de matcheo en píxeles', default=10)
args = parser.parse_args()
COLUMNS = ['image_name','x','y','r','detection','track_id','label']

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


def to_cloud(frame_detections_dict, x=0, y=0):
    # print(frame_detections_dict)
    vector = [[det[0]+x, det[1]+y,  0.] for det in frame_detections_dict.values()]
    vector = np.array(vector)
    return conform_point_cloud(vector)



def main(args):
    # print(args.input)
    radius=args.radius
    detections_file = open(args.input, 'r')
    detections = json.load(detections_file)
    detections_file.close()
    json_path, json_name = os.path.split(args.input)
    video_name, _ = os.path.splitext(json_name)
    tracker = Tracker(COLUMNS, video_name, args)
    print(args.video_path)
    for i in range(len(detections)-1):
        if i==0:
            previous = detections[str(i)]
            tracker.init_ids(previous)
            continue

        tracker.frame += 1
        current = detections[str(i)]
        previous = detections[str(i-1)]
        previus_len = len(previous)
        current_len = len(current)
        current_cloud = to_cloud(current)
        previous_cloud = to_cloud(previous)
        icp = o3d.pipelines.registration.registration_icp(previous_cloud, current_cloud, radius)
        correspondence_set = np.asarray(icp.correspondence_set)
        print(f'fitntess: {icp.fitness}; len prev: {previus_len};en curr {current_len}, matcheos: {len(correspondence_set)}')
        fitness = icp.fitness
        if icp.fitness <= 0.85:
            for px_shift in [20, 25, 30, 35, 40, 45]:
                for x,y in [(px_shift, 0), (px_shift, px_shift), (0, px_shift), (-px_shift, 0), (-px_shift, -px_shift), (0, -px_shift)]:
                    previous_copy = copy.deepcopy(previous)
                    previous_copy_cloud_aux = to_cloud(previous_copy, x, y)
                    icp2 = o3d.pipelines.registration.registration_icp(previous_copy_cloud_aux, current_cloud, radius)

                    if icp.fitness < icp2.fitness:
                        icp = icp2
                        previous_copy_cloud = copy.deepcopy(previous_copy_cloud_aux)

            if icp.fitness == fitness:
                previous_copy_cloud = previous_cloud
            correspondence_set = np.asarray(icp.correspondence_set)
            print(f'  fitntess: {icp.fitness}; len prev: {previus_len};en curr {current_len}, matcheos: {len(correspondence_set)}')

            # previous_matched_idx = correspondence_set[:, 0]
            # current_matched_idx = correspondence_set[:, 1]
            # previous_copy_cloud.transform(icp.transformation)
            # colors = np.zeros((len(previous_copy_cloud.points), 3))
            # colors[:] = [0, 1, 0]
            # colors[previous_matched_idx, :] = [1, 0, 0]
            # previous_copy_cloud.colors = o3d.utility.Vector3dVector(colors)
            # colors = np.zeros((len(current_cloud.points), 3))
            # colors[:] = [1, 0, 1]
            # colors[current_matched_idx, :] = [1, 1, 1]
            # current_cloud.colors = o3d.utility.Vector3dVector(colors)
            # point_cloud_viewer([current_cloud, previous_copy_cloud])
        tracker.update_ids(correspondence_set, current, previous)


    tracker.write_results(args.output)

if __name__ == '__main__':
    main(args)