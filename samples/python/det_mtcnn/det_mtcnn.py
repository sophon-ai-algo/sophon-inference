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
import cv2
import sys
import os
import argparse
import numpy as np
from sophon import sail
from processor import PreProcessor
from processor import PostProcessor

def run_pnet(engine, preprocessor, postprocessor, image):
  """ Run PNet, the first stage of MTCNN.

  Args:
    engine: Engine instance
    preprocessor: PreProcessor instance
    postprocessor: PostProcessor instance
    image: Input image

  Returns:
    Detected boxes
  """
  graph_name = 'PNet'
  input_name = engine.get_input_names(graph_name)[0];
  boxes = np.zeros((0, 9), np.float32)
  height = image.shape[0]
  width = image.shape[1]
  scales = preprocessor.generate_scales(height, width)
  for scale in scales:
    scaled_h = int(np.ceil(height * scale))
    scaled_w = int(np.ceil(width * scale))
    input_data = preprocessor.pnet_process(image, scaled_h, scaled_w)
    output_tensors = engine.process(graph_name, {input_name: input_data})
    candidates = postprocessor.pnet_process_per_scale(output_tensors, scale)
    if candidates is not None and len(candidates) > 0:
      boxes = np.concatenate((boxes, candidates), axis=0)
  boxes = postprocessor.pnet_process(boxes)
  boxes_num = 0 if boxes is None else boxes.shape[0]
  print("Box number detected by PNet: {}".format(boxes_num));
  return boxes

def run_rnet(engine, preprocessor, postprocessor, boxes, image):
  """ Run RNet, the second stage of MTCNN.

  Args:
    engine: Engine instance
    preprocessor: PreProcessor instance
    postprocessor: PostProcessor instance
    boxes: Detected boxes by PNet
    image: Input image

  Returns:
    Detected boxes
  """
  graph_name = 'RNet'
  input_name = engine.get_input_names(graph_name)[0];
  input_shape = engine.get_input_shape(graph_name, input_name);
  height = input_shape[2]
  width = input_shape[3]
  data = preprocessor.rnet_process(image, boxes, height, width)
  input_shapes = {input_name: input_shape}
  box_num = boxes.shape[0]
  num_fix = input_shape[0]
  num_done = 0
  while num_done < box_num:
    if (num_done + num_fix) > box_num:
      num_fix = box_num - num_done
    num_work = num_fix
    input_shapes[input_name][0] = num_work
    input_tensors = {input_name: data[num_done:num_done + num_work, :, :, :]}
    output_tensors = engine.process(graph_name, input_tensors)
    if num_done == 0:
      output = output_tensors
    else:
      for key in output:
        output[key] = np.concatenate((output[key],
                                      output_tensors[key].copy()), axis=0)
    num_done += num_work
  for key in output:
    output[key] = np.reshape(output[key], output[key].shape[0:2])
  boxes = postprocessor.rnet_process(output, boxes)
  boxes_num = 0 if boxes is None else boxes.shape[0]
  print("Box number detected by RNet: {}".format(boxes_num));
  return boxes

def run_onet(engine, preprocessor, postprocessor, boxes, image):
  """ Run ONet, the second stage of MTCNN.

  Args:
    engine: Engine instance
    preprocessor: PreProcessor instance
    postprocessor: PostProcessor instance
    boxes: Detected boxes by ONet
    image: Input image

  Returns:
    Detected boxes
  """
  graph_name = 'ONet'
  input_name = engine.get_input_names(graph_name)[0];
  input_shape = engine.get_input_shape(graph_name, input_name);
  height = input_shape[2]
  width = input_shape[3]
  data = preprocessor.onet_process(image, boxes, height, width)
  input_shapes = {input_name: input_shape}
  box_num = boxes.shape[0]
  num_fix = input_shape[0]
  num_done = 0
  while num_done < box_num:
    if (num_done + num_fix) > box_num:
      num_fix = box_num - num_done
    num_work = num_fix
    input_shapes[input_name][0] = num_work
    input_tensors = {input_name: data[num_done:num_done + num_work, :,  :, :]}
    output_tensors = engine.process(graph_name, input_tensors)
    if num_done == 0:
      output = output_tensors
    else:
      for key in output:
        output[key] = np.concatenate((output[key],
                                      output_tensors[key].copy()), axis=0)
    num_done += num_work
  for key in output:
    output[key] = np.reshape(output[key], output[key].shape[0:2])
  boxes = postprocessor.onet_process(output, boxes)
  boxes_num = 0 if boxes is None else boxes.shape[0]
  print("Box number detected by ONet: {}".format(boxes_num));
  return boxes

def print_result(boxes, tpu_id):
  """ Print bounding boxes of detected faces.

  Args:
    boxes: Detected bounding boxes
    tpu_id: TPU ID

  Returns:
    None
  """
  if boxes is None or len(boxes) == 0:
    print("No face was detected in this image!");
    return
  print("---------  MTCNN DETECTION RESULT ON TPU {} ---------".format(tpu_id));
  message = "Face {} Box: [{}, {}, {}, {}], Score: {:.6f}"
  for i in range(boxes.shape[0]):
    x = int(boxes[i, 1]) if boxes[i, 1] > 0 else 0
    y = int(boxes[i, 0]) if boxes[i, 0] > 0 else 0
    width = int(boxes[i, 3] - boxes[i, 1])
    height = int(boxes[i, 2] - boxes[i, 0])
    print(message.format(i, x, y, width, height, boxes[i, 4]))

def inference(bmodel_path, input_path, loops, tpu_id, compare_path):
  """ Load a bmodel and do inference.

  Args:
   bmodel_path: Path to bmodel
   input_path: Path to input file
   loops: Number of loops to run
   tpu_id: ID of TPU to use
   compare_path: Path to correct result file

  Returns:
    True for success and False for failure
  """
  # init Engine to load bmodel and allocate input and output tensors
  engine = sail.Engine(bmodel_path, tpu_id, sail.SYSIO)
  # init preprocessor and postprocessor
  preprocessor = PreProcessor([127.5, 127.5, 127.5], 0.0078125)
  postprocessor = PostProcessor([0.5, 0.3, 0.7])
  reference = postprocessor.get_reference(compare_path)
  status = True
  # pipeline of inference
  for i in range(loops):
    # read image
    image = cv2.imread(input_path)
    image = cv2.transpose(image)
    # run PNet, the first stage of MTCNN
    boxes = run_pnet(engine, preprocessor, postprocessor, image)
    if boxes is not None and len(boxes) > 0:
      # run RNet, the second stage of MTCNN
      boxes = run_rnet(engine, preprocessor, postprocessor, boxes, image)
      if boxes is not None and len(boxes) > 0:
        # run ONet, the third stage of MTCNN
        boxes = run_onet(engine, preprocessor, postprocessor, boxes, image)
    # print detected result
    if postprocessor.compare(reference, boxes, i):
      print_result(boxes, tpu_id)
    else:
      status = False
      break
  return status

if __name__ == '__main__':
  """An example of inference for dynamic model, whose input shapes may change.

     There are 3 graphs in the MTCNN model: PNet, RNet and ONet. Input height
     and width may change for Pnet while input batch_szie may change for RNet
     and Onet.
  """
  PARSER = argparse.ArgumentParser(description='det_mtcnn')
  PARSER.add_argument('--bmodel', default='', required=True)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--loops',  default=1, type=int, required=False)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  PARSER.add_argument('--compare', default='', required=False)
  ARGS = PARSER.parse_args()
  if not os.path.isfile(ARGS.input):
    print("Error: {} not exists!".format(ARGS.input))
    sys.exit(-2)
  status = inference(ARGS.bmodel, ARGS.input, \
                     ARGS.loops, ARGS.tpu_id, ARGS.compare)
  sys.exit(0 if status else -1)
