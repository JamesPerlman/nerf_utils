import argparse
import copy
import json
import numpy as np

from pathlib import Path

parser = argparse.ArgumentParser(description='Reorient transforms.json')

parser.add_argument('--input', required=True, help='Instant-ngp transforms.json whose frames you want to change.')
parser.add_argument('--matrix', required=True, help='A world transform matrix.json file to reorient the scene')
parser.add_argument('--output', help='Where to put the manipulated transforms.json - if not given, the input file will be replaced.')

args = parser.parse_args()

if args.output == None:
    args.output = args.input

input_path = Path(args.input)
output_path = Path(args.output)
matrix_path = Path(args.matrix)

def read_json_file(file_path: Path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def write_json_file(data: dict, file_path: Path):
    with open(file_path, 'w+') as json_file:
        json.dump(data, json_file, indent=4)

input_json = read_json_file(input_path)
matrix_json = read_json_file(matrix_path)
output_json = copy.deepcopy(input_json)

world_matrix = np.array(matrix_json['transform_matrix'])

def transformed_frame(frame_json: dict, matrix: np.ndarray):
    output_frame = copy.deepcopy(frame_json)
    output_frame['transform_matrix'] = np.matmul(matrix, np.array(frame_json['transform_matrix'])).tolist()
    return output_frame

# write output json
output_json['frames'] = [transformed_frame(frame, world_matrix) for frame in input_json['frames']]

write_json_file(output_json, output_path)
