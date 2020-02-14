""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import time
import tensorflow as tf
import numpy as np
from google.protobuf import text_format


def node_def_add_node(sss):
  """ add node{} to a string of an op.

  Args:
    sss: original string of an op.

  Returns:
    new string with node{} added.
  """
  sss = sss.replace('\n', '\n\t')
  sss = "node {\n\t" + sss
  sss = sss[0:len(sss)-1] + '}\n'
  return sss

def node_def_remove_colocations(sss):
  """ remove colocations in a node_def.

  Args:
    sss: original node_def of an op.

  Returns:
    new string of node_def.
  """
  idx = sss.find('loc:@')
  if idx < 0:
    return sss
  idx1 = sss.find('"', idx)
  to_remove = sss[idx:idx1]
  sss = sss.replace(to_remove, '')
  return sss

def if_this_input_can_remove(tfgraph, iop):
  """ Judge if this op can remove,
      or, to say, an constant value
  """
  operator = tfgraph.get_operation_by_name(iop)
  if operator.type == 'Const':
    return True, [iop]
  if operator.type == 'Identity':
    if len(operator.inputs) == 1:
      inputop = operator.inputs[0].op
      if inputop.type == 'Const':
        return True, [iop, inputop.name]
  return False, None

def node_def_remove_control_inputs(sss):
  """ remove controlinputs in a node_def.

  Args:
    sss: original node_def of an op.

  Returns:
    new string of node_def.
  """
  idx = sss.find('input: "^')
  while idx > 0:
    idx_e = sss.find('"', idx + 10)
    tmpstr = sss[idx:(idx_e+1)]
    sss = sss.replace(tmpstr, '')
    idx = sss.find('input: "^')
  return sss

def node_def_remove_inputs(sss):
  """ remove inputs in a node_def.

  Args:
    sss: original node_def of an op.

  Returns:
    new string of node_def.
  """
  idx = sss.find('input: "')
  while idx > 0:
    idx_e = sss.find('"', idx + 9)
    tmpstr = sss[idx:(idx_e+1)]
    sss = sss.replace(tmpstr, '')
    idx = sss.find('input: "')
  return sss

def just_write_text(sss, pb_path, pb_name):
  """ only for debug
  """
  tid = sss.find('tensor_content:')
  while tid > 0:
    eid = sss.find('}', tid)
    sss = sss[:tid] + sss[eid:]
    tid = sss.find('tensor_content:')
  with open(os.path.join(pb_path, pb_name+'txt'), 'w+') as fff:
    fff.write(sss)

def save_str_to_pb(sss, pb_path, pb_name):
  """ Save node_def string to a model, ends with ".pb".
  """
  #just_write_text(sss, pb_path, pb_name)
  graph_def = tf.GraphDef()
  text_format.Merge(sss, graph_def)
  tf.train.write_graph(graph_def, pb_path, pb_name, as_text=False)

def infer_tensorflow_graph(graph_path, input_names, input_values, \
          output_names, warm_loop=0, infer_loop=1, with_flops=False):
  """ infer a tensorflow graph.
  """
  graph = tf.Graph()
  with graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph_path, 'rb') as fid:
      serialized_graph = fid.read()
      graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(graph_def, name='')
      with tf.Session() as sess:
        for _ in range(warm_loop):
          rets = sess.run(output_names, \
                    feed_dict=dict(zip(input_names, input_values)))
        time_1 = time.time()
        for _ in range(infer_loop):
          rets = sess.run(output_names, \
                    feed_dict=dict(zip(input_names, input_values)))
        time_2 = time.time()
        flop_value = None
        if with_flops:
          flops = tf.profiler.profile(graph, \
                  options=tf.profiler.ProfileOptionBuilder.float_operation())
          flop_value = flops.total_float_ops
  return rets, (time_2 - time_1)/infer_loop, flop_value

def get_all_ops_from_model_path(model_path):
  """ get all ops and the graph of a tensorflow model.
  """
  assert model_path.endswith(".pb")
  graph = tf.Graph()
  with graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, "rb") as fid:
      serialized_graph = fid.read()
      graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(graph_def, name='')
  ops = graph.get_operations()
  return ops, graph

def get_graph(model_path, device):
  """ get graph of a tensorflow model
  """
  assert model_path.endswith(".pb")
  graph = tf.Graph()
  with graph.as_default():
    with tf.device(device):
      graph_def = tf.GraphDef()
      with tf.gfile.GFile(model_path, "rb") as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')
  return graph


def tensorflow_infer_outputs(model, input_dict, output_names):
  """ infer tensorflow graph and get its result.
  """
  if isinstance(model, str):
  #if type(model) == type(str()):
    _, graph = get_all_ops_from_model_path(model)
  else:
    graph = model
  with graph.as_default():
    with tf.Session() as sess:
      ret = sess.run(output_names, input_dict)
  return ret

def get_tensor_shape_in_graph(graph, start_tensors, \
                              start_shapes, out_tensor_name):
  """ infer a tensorflow graph, get the shape of the querying tensor.
  """
  operator = graph.\
      get_operation_by_name(out_tensor_name[:out_tensor_name.find(':')])
  if len(operator.outputs) < 1:
    return tuple([-1])
  inputs = []
  for s_tensor in start_tensors:
    inputs.append(graph.get_tensor_by_name(s_tensor))
  values = []
  for s_shape in start_shapes:
    values.append(np.ones(s_shape))
  with graph.as_default():
    with tf.Session() as sess:
      ret = sess.run(out_tensor_name, feed_dict=dict(zip(inputs, values)))
  # it's a tuple
  return ret.shape

def get_flops(model_path):
  """ get flops of a model

  Args:
    model_path: A string, path of the model.

  Returns:
    An integer, flops of this model

  """
  _, graph = get_all_ops_from_model_path(model_path)
  flops = tf.profiler.profile(graph, \
            options=tf.profiler.ProfileOptionBuilder.float_operation())
  return flops.total_float_ops
