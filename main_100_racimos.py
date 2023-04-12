import main
import os

def make_dir(path, name):
    if not os.path.isdir(path+name):
        os.mkdir(path+name)

videos_root = 'F:/Documentos/Dharma/captura_2_2023/capturas/'
input_root = 'F:/Documentos/Dharma/captura_2_2023/detections/'
output_root = 'F:/Documentos/Dharma/captura_2_2023/tracked/'
path_directory_list =['freestyle/', '180/', 'verticales/']

class Args:
    def __init__(self):
        self.draw_tracking = True
        self.draw_circles = False
        self.radius = 12
args = Args()

for directory in path_directory_list:
    videos_root_path = videos_root + directory
    input_path = input_root + directory
    inputs_file = input_path + 'input.txt'
    video_inputs_file = videos_root_path + 'input.txt'
    json_file = open(inputs_file, 'r')
    videos_file = open(video_inputs_file, 'r')
    for jsf, vf in zip(json_file, videos_file):
        args.input = input_path + jsf[:-1]
        args.video_path = videos_root_path + vf[:-1]
        make_dir(output_root+directory, jsf[:-5])
        args.output = output_root+directory+jsf[:-6]
        main.main(args)
