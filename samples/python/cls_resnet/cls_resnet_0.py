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
import argparse
import numpy as np
from sophon import sail
from processor import preprocess
from processor import postprocess
from processor import get_reference
from processor import compare

def inference(bmodel_path, input_path, loops, tpu_id, compare_path):
  """ Do inference of a model in a thread.

  Args:
    bmodel_path: Path to bmodel
    input_path: Path to input image.
    loops: Number of loops to run
    compare_path: Path to correct result file
    status: Status of comparison

  Returns:
    True for success and False for failure.
  """
  # init Engine to load bmodel and allocate input and output tensors
  engine = sail.Engine(bmodel_path, tpu_id, sail.SYSIO)
  # get model info
  # only one model loaded for this engine
  # only one input tensor and only one output tensor in this graph
  graph_name = engine.get_graph_names()[0]
  input_name = engine.get_input_names(graph_name)[0]
  input_shape = engine.get_input_shape(graph_name, input_name)
  output_name = engine.get_output_names(graph_name)[0]
  output_shape = engine.get_output_shape(graph_name, output_name)
  out_dtype = engine.get_output_dtype(graph_name, output_name);
  reference = get_reference(compare_path)
  compare_type = 'fp32_top5' if out_dtype == sail.BM_FLOAT32 else 'int8_top5'
  # pipeline of inference
  for i in range(loops):
    # read image and preprocess
    image = preprocess(input_path).astype(np.float32)
    # inference with fp32 input and output
    # data scale(input: fp32 to int8, output: int8 to fp32) is done inside
    # for int8 model
    output = engine.process(graph_name, {input_name:image})
    # postprocess
    result = postprocess(output[output_name])
    # print result
    print("Top 5 of loop {}: {}".format(i, result[1]['top5_idx'][0]))
    if not compare(reference, result[1]['top5_idx'][0], compare_type):
      return False
  return True

if __name__ == '__main__':
  """ It is the simplest case for inference of one model on one TPU.
  """
  PARSER = argparse.ArgumentParser(description='cls_resnet')
  PARSER.add_argument('--bmodel', default='', required=True)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--loops', default=1, type=int, required=False)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  PARSER.add_argument('--compare', default='', required=False)
  ARGS = PARSER.parse_args()
  status = inference(ARGS.bmodel, ARGS.input,
                     ARGS.loops, ARGS.tpu_id, ARGS.compare)
  sys.exit(0 if status else -1)
