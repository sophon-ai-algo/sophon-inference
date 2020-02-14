"""Check tensorflow split script."""
import os
import glob
import json
import argparse
import time
import tensorflow as tf
import numpy as np
from .runner.tensorflow_runner import TensorflowRunner as RN

def parse_args():
  """argument parser"""
  parser = argparse.ArgumentParser(description='performance of auto_runner.')
  parser.add_argument('--submodel_path', type=str, default='',
                      required=True,
                      help="Tensorflow sub model path")
  parser.add_argument('--warm_loop_num', type=int, default=10, required=False,
                      help="num of warm-up batches.")
  parser.add_argument('--test_loop_num', type=int, default=100, required=False,
                      help="num of test batches.")
  return parser.parse_args()

def parse_json(graph_dict):
  """graph infos parse"""
  tensor_dict = graph_dict['tensors']
  input_dict = {}
  output_dict = {}
  for key in tensor_dict:
    if tensor_dict[key]['attr'] == 'input':
      input_dict[key] = \
        np.random.random(tensor_dict[key]['shape']).astype(np.float32)
  graphs = graph_dict['graphs']
  devices = []
  for g in graphs:
    devices.append(g['device'])
  return input_dict, devices

if __name__ == '__main__':
  args = parse_args()
  # check model path
  if not os.path.exists(args.submodel_path):
    raise FileNotFoundError("input path is not valid.")
  if args.warm_loop_num <= 0 or args.test_loop_num <= 0:
    raise RuntimeError("loop num should be more than 0.")
  json_path = os.path.join(args.submodel_path, 'graph_infos.json')
  if not os.path.exists(json_path):
    raise FileNotFoundError("No graph_infos.json under input path.")
  # check json file && open graph_infos.json
  with open(json_path) as load_f:
    load_dict = json.load(load_f)
  # get input && output tensors
  input_tensors, devices = parse_json(load_dict)
  
  load_start = time.time()
  model = RN(args.submodel_path)
  load_end = time.time()
  
  warm_start = time.time()
  for _ in range(args.warm_loop_num):
    times = model.infer_time(input_tensors)
  warm_end = time.time()

  time_array = np.zeros(len(times))
  for _ in range(args.test_loop_num):
    times = model.infer_time(input_tensors)
    time_array += np.array(times)
  time_array = 1000.0 * time_array / args.test_loop_num
  
  print("--------------------performance--------------------")
  print("Loading time(ms) is: {}".format((load_end - load_start) * 1000.0))
  print("Average inference time(ms) is: {}".format(time_array[-1]))
  for idx, dev in enumerate(devices):
    print("  Part {0} on {1}, inference time(ms) is: {2}".format(idx, dev, time_array[idx]))
