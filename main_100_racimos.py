import main
import os
import time
def make_dir(path, name):
    if not os.path.isdir(path+name):
        os.mkdir(path+name)

videos_root = 'F:/Documentos/Dharma/captura_2_2023/capturas/'
videos_root = None
input_root = 'F:/Documentos/Dharma/captura_2_2023/detections/'
output_root = 'F:/Documentos/Dharma/captura_2_2023/tracked/'
path_directory_list =['180/','freestyle/', 'verticales/']

class Args:
    def __init__(self):
        self.draw_tracking = False
        self.draw_circles = False
        self.radius = 12
args = Args()

for directory in path_directory_list:
    if  args.draw_tracking:
        videos_root_path = videos_root + directory
        video_inputs_file = videos_root_path + 'input.txt'
        videos_file = open(video_inputs_file, 'r')
    else:
        videos_root_path =  None
        video_inputs_file = None
        videos_file = None
    input_path = input_root + directory
    inputs_file = input_path + 'input.txt'
    json_file = open(inputs_file, 'r')

    if videos_file is None:
        for jsf in json_file:
            ini = time.time()
            args.input = input_path + jsf[:-1]
            args.video_path = None
            make_dir(output_root+directory, jsf[:-5])
            args.output = output_root+directory+jsf[:-6]
            main.main(args)
            end = time.time()
            print(f'time: {end-ini:2.2}')
    else:
        for jsf, vf in zip(json_file, videos_file):
            ini = time.time()
            args.input = input_path + jsf[:-1]
            args.video_path = videos_root_path + vf[:-1]
            make_dir(output_root+directory, jsf[:-5])
            args.output = output_root+directory+jsf[:-6]
            main.main(args)
            end = time.time()
            print(f'time: {end-ini:2.2}')