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
import sys
import os
import argparse
import threading
import json
import numpy as np
from sophon import sail
from processor import preprocess
from processor import postprocess
from processor import get_reference
from processor import compare

def thread_load(thread_id, engine, bmodel_path):
  """ Load a model in a thread.

  Args:
    thread_id: ID of the thread.
    engine: An sail.Engine instance.
    bmodel_path: Path to bmodel.

  Returns:
    None.
  """
  ret = engine.load(bmodel_path)
  if ret == 0:
    graph_name = engine.get_graph_names()[-1]
    print("Thread {} load {} successfully.".format(thread_id, graph_name))

def thread_infer(thread_id, graph_name, engine, \
                 input_path, loops, compare_path, status):
  """ Do inference of a model in a thread.

  Args:
    thread_id: ID of the thread.
    graph_name: Name of the model or graph.
    engine: An sail.Engine instance.
    input_path: Path to input image.
    loops: Number of loops to run
    compare_path: Path to correct result file
    status: Status of comparison

  Returns:
    None.
  """
  input_name = engine.get_input_names(graph_name)[0]
  input_shape = engine.get_input_shape(graph_name, input_name)
  output_name = engine.get_output_names(graph_name)[0]
  output_shape = engine.get_output_shape(graph_name, output_name)
  in_dtype = engine.get_input_dtype(graph_name, input_name);
  out_dtype = engine.get_output_dtype(graph_name, output_name);
  # get handle to create input and output tensors
  handle = engine.get_handle()
  input = sail.Tensor(handle, input_shape, in_dtype, True, True)
  output = sail.Tensor(handle, output_shape, out_dtype, True, True)
  input_tensors = {input_name:input}
  ouptut_tensors = {output_name:output}
  # set io_mode
  engine.set_io_mode(graph_name, sail.SYSIO)
  reference = get_reference(compare_path)
  compare_type = 'fp32_top5' if out_dtype == sail.BM_FLOAT32 else 'int8_top5'
  # pipeline of inference
  for i in range(loops):
    # read image and preprocess
    image = preprocess(input_path).astype(np.float32)
    # scale input data if input data type is int8 or uint8
    if in_dtype == sail.BM_FLOAT32:
      input_tensors[input_name].update_data(image)
    else:
      scale = engine.get_input_scale(graph_name, input_name)
      input_tensors[input_name].scale_from(image, scale)
    # inference
    engine.process(graph_name, input_tensors, ouptut_tensors)
    # scale output data if output data type is int8 or uint8
    if out_dtype == sail.BM_FLOAT32:
      output_data = output.asnumpy()
    else:
      scale = engine.get_output_scale(graph_name, output_name)
      output_data = output.scale_to(scale)
    # postprocess
    result = postprocess(output_data)
    # print result
    print("Top 5 for {} of loop {} in thread {} on tpu {}: {}".format(\
        graph_name, i, thread_id, engine.get_device_id(), \
        result[1]['top5_idx'][0]))
    if not compare(reference, result[1]['top5_idx'][0], compare_type):
      status[thread_id] = False
      return
  status[thread_id] = True

def main():
  """ An example shows inference of multi-models in multi-threads on one TPU.
  """
  # init Engine
  engine = sail.Engine(ARGS.tpu_id)
  # create threads for loading bmodel, you can also load in main thread
  thread_num = len(ARGS.bmodel)
  threads = list()
  # load bmodel without builtin input and output tensors
  # each thread manage its input and output tensors
  for i in range(thread_num):
    threads.append(threading.Thread(target=thread_load,
                                    args=(i, engine, ARGS.bmodel[i])))
  for i in range(thread_num):
    threads[i].start()
  for i in range(thread_num):
    threads[i].join()
  threads.clear()
  status = [None] * thread_num
  graph_names = engine.get_graph_names()
  # create threads for inference, and each thread processes a model
  for i in range(thread_num):
    threads.append(threading.Thread(target=thread_infer,
                                    args=(i, graph_names[i], engine, \
                                          ARGS.input, ARGS.loops, \
                                          ARGS.compare, status)))
  for i in range(thread_num):
    threads[i].start()
  for i in range(thread_num):
    threads[i].join()
  # check status
  for stat in status:
    if not stat:
      sys.exit(-1)
  sys.exit(0)

if __name__ == '__main__':
  PARSER = argparse.ArgumentParser(description='cls_resnet')
  PARSER.add_argument('--bmodel', action='append', type=str, required=True)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--loops', default=1, type=int, required=False)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  PARSER.add_argument('--compare', default='', required=False)
  ARGS = PARSER.parse_args()
  if not os.path.isfile(ARGS.input):
    print("Error: {} not exists!".format(ARGS.input))
    sys.exit(-2)
  main()
