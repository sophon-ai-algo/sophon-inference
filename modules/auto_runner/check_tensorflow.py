"""Check tensorflow split script."""
import os
import glob
import json
import argparse
import tensorflow as tf
import numpy as np
from sophon.auto_runner.api import load, load_from_zip, infer
from sophon.utils.functions import read_graph_infos_from_zip

def parse_args():
  """argument parser"""
  parser = \
    argparse.ArgumentParser(description='Check tensorflow split correctness.')
  parser.add_argument('--tfmodel_path', type=str, default='',
                      required=True,
                      help="Tensorflow pb model path")
  parser.add_argument('--submodel_path', type=str, default='',
                      required=True,
                      help="Tensorflow sub model path")
  parser.add_argument('--use_cmodel', type=int, default=0,
                      required=False,
                      help="0 or 1, if running on cmodel mode.")
  parser.add_argument('--from_zip', type=int, default=0,
                      required=False,
                      help="0 or 1, if load from zip")
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
    if tensor_dict[key]['attr'] == 'output':
      output_dict[key] = tensor_dict[key]['shape']
  return input_dict, output_dict

def compare_data(x_data, y_data, eps):
  """ Compare two numpy.ndarray.
  Args:
  -----
    x: numpy.ndarray.
    y: numpy.ndarray.
    eps: Threshold.
  Return:
  -------
    True for equal and False for not.
  """
  a_data = x_data.flatten()
  b_data = y_data.flatten()
  if a_data.shape != b_data.shape:
    print("shape not match")
    print(x_data.shape, " vs ", y_data.shape)
    return False
  for i, _ in enumerate(a_data):
  #for i in range(len(a_data)):
    if abs(a_data[i]) > 1.0:
      if abs((a_data[i] - b_data[i]) / a_data[i]) > eps:
        print("{} vs {}".format(a_data[i], b_data[i]))
        return False
    elif abs(a_data[i] - b_data[i]) > eps:
      print("{} vs {}".format(a_data[i], b_data[i]))
      return False
  return True

def compare_outputs(raw_out, new_out, output_names):
  """ Compare output tensors between the raw graph and subgraphs.
  Args:
  -----
    raw_out: Outputs of the raw graph. Format: {output_name, numpy.ndarray}
    new_out: Outputs of subgraphs. Format: {output_name: numpy.ndarray}
    output_names: A list contains names of final output intensors.
  Return:
  -------
    True for equal and False for not.
  """
  eps = 1e-4
  ret = True
  if len(output_names) == 1:
    raw_data = raw_out[output_names[0]]
    new_data = new_out[output_names[0]]
    ret = compare_data(raw_data, new_data, eps)
    if not ret:
      print("Compare failed for tenssor: ", output_names[0])
  else:
    for i, _ in enumerate(output_names):
    #for i in range(len(output_names)):
      raw_data = raw_out[output_names[i]]
      new_data = new_out[output_names[i]]
      ret = compare_data(raw_data, new_data, eps)
      if not ret:
        print("Compare failed for tenssor: ", output_names[i])
  return ret

def run_cpu(model_path, input_tensors, output_tensors):
  """run tensorflow model with pure cpu"""
  cpu_graph = tf.Graph()
  with cpu_graph.as_default():
    inner_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
      inner_graph_def.ParseFromString(fid.read())
      tf.import_graph_def(inner_graph_def, name='')
    with tf.Session() as sess:
      cpu_input_tensor = {}
      for key in input_tensors:
        inner_tensor = tf.get_default_graph().get_tensor_by_name(key)
        cpu_input_tensor[inner_tensor] = input_tensors[key]
      cpu_output_tensor = {}
      for key in output_tensors:
        cpu_output_tensor[key] = tf.get_default_graph().get_tensor_by_name(key)
      cpu_results = sess.run(cpu_output_tensor, feed_dict=cpu_input_tensor)
  return cpu_results

def run_real(submodel_path, input_tensors, use_cmodel, from_zip):
  """run tensorflow model with tpu&&cpu"""
  if from_zip == 1:
    model = load_from_zip(submodel_path, use_cmodel)
  else:
    model = load(submodel_path, use_cmodel=use_cmodel)
  return infer(model, input_tensors)

if __name__ == '__main__':
  args = parse_args()
  # check model path
  if not (os.path.exists(args.tfmodel_path) \
      and os.path.exists(args.submodel_path)):
    raise FileNotFoundError("input path is not valid")
  # check json file && open graph_infos.json
  if args.from_zip == 1:
    load_dict = json.loads(read_graph_infos_from_zip(args.submodel_path))
  else:
    json_list = glob.glob(args.submodel_path + '*.json')
    assert(len(json_list) == 1)
    with open(json_list[0]) as load_f:
      load_dict = json.load(load_f)
  # get input && output tensors
  input_tensors, output_tensors = parse_json(load_dict)
  # get pure cpu results && tpu+cpu results
  cpu_results = run_cpu(args.tfmodel_path, input_tensors, output_tensors)
  real_results = run_real(args.submodel_path, input_tensors, args.use_cmodel, args.from_zip)
  # cmp pure with tpu+cpu
  ret = compare_outputs(cpu_results, real_results, [ name for name in output_tensors])
  if ret:
    print("\x1b[32m" + "+++" + "compared pure cpu with tpu&&cpu success!" + "+++" + "\x1b[0m")