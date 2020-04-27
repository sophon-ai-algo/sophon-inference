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
from __future__ import division
import sys
import os
import argparse
import json
import cv2
import numpy as np
import sophon.sail as sail

def preprocess(image, detected_size):
  """ Preprocessing of a frame.

  Args:
    image : np.array, input image
    detected_size : list, yolov3 detection input size

  Returns:
    Preprocessed data.
  """
  img_h, img_w = image.shape[0], image.shape[1]
  net_h, net_w = detected_size[0], detected_size[1]
  new_w = int(img_w * min(net_w / img_w, net_h / img_h))
  new_h = int(img_h * min(net_w / img_w, net_h / img_h))
  resized_image = cv2.resize(image, (new_w, new_h), \
                             interpolation=cv2.INTER_LINEAR)
  # canvas
  canvas = np.full((net_w, net_h, 3), 127.5)
  canvas[(net_h - new_h) // 2:(net_h - new_h) // 2 + new_h,\
    (net_w - new_w) // 2:(net_w - new_w) // 2 + new_w, :]\
    = resized_image
  return canvas[:, :, ::-1].transpose([2, 0, 1]) / 255.0

def postprocess(output, image, detected_size, threshold):
  """ Postprocessing of YOLOv3.

  Args:
    output: dict, inference output
    image: np.array, detected image
    detected_size: list, detected size
    threshold: float,score threshold

  Returns:
    Detected boxes, class ids, probability
  """
  detections = output['detection_out'][0][0]
  img_h, img_w = image.shape[0], image.shape[1]
  net_h, net_w = detected_size[0], detected_size[1]
  new_w = (img_w * min(net_w / img_w, net_h / img_h))
  new_h = (img_h * min(net_w / img_w, net_h / img_h))
  scale_w = new_w / net_w
  scale_h = new_h / net_h
  cls=[]
  probs=[]
  bboxes=[]
  for det in detections:
    if det[2] > threshold:
      cls.append((int)(det[1]))
      probs.append(det[2])
      x = (det[3] - (net_w - new_w) / 2. / net_w) / scale_w * img_w
      y = (det[4] - (net_h - new_h) / 2. / net_h) / scale_h * img_h
      w = det[5] * img_w / scale_w
      h = det[6] * img_h / scale_h
      x1 = x - w / 2.
      y1 = y - h / 2.
      x2 = x + w / 2.
      y2 = y + h / 2.
      bbox = []
      bbox.append(int(x1))
      bbox.append(int(y1))
      bbox.append(int(x2))
      bbox.append(int(y2))
      bboxes.append(bbox)
  return bboxes, cls, probs

def get_reference(compare_path):
  """ Get correct result from given file.
  Args:
    compare_path: Path to correct result file

  Returns:
    Correct result.
  """
  if compare_path:
    with open(compare_path, 'r') as f:
      reference = json.load(f)
      return reference
  return None

def compare(reference, bboxes, classes, probs, loop_id):
  """ Compare result.
  Args:
    reference: Correct result
    result: Output result
    loop_id: Loop iterator number

  Returns:
    True for success and False for failure
  """
  if not reference or loop_id > 0:
    return True
  detected_num = len(classes)
  reference_num = len(reference["category"])
  if (detected_num != reference_num):
    message = "Expected deteted number is {}, but detected {}!"
    print(message.format(reference_num, detected_num))
    return False
  ret = True
  scores = ["{:.8f}".format(p) for p in probs]
  message = "Category: {}, Score: {}, Box: {}"
  fail_info = "Compare failed! Expect: " + message
  ret_info = "Result Box: " + message
  for i in range(detected_num):
    if classes[i] != reference["category"][i] or \
        scores[i] != reference["score"][i] or \
        bboxes[i] != reference["box"][i]:
      print(fail_info.format(reference["category"][i], reference["score"][i], \
                             reference["box"][i]))
      print(ret_info.format(classes[i], scores[i], bboxes[i]))
      ret = False
  return ret

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
  # set configurations
  load_from_file = True
  detected_size = (416, 416)
  threshold = 0.5
  nms_threshold = 0.45
  num_classes = 80
  cap = cv2.VideoCapture(input_path)
  # init Engine and load bmodel
  if load_from_file:
    # load bmodel from file
    net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
  else:
    # simulate load bmodel from memory
    f = open(file=bmodel_path, mode='rb')
    bmodel = f.read()
    f.close()
    net = sail.Engine(bmodel, len(bmodel), tpu_id, sail.IOMode.SYSIO)
  # get model info
  graph_name = net.get_graph_names()[0]
  input_name = net.get_input_names(graph_name)[0]
  reference = get_reference(compare_path)
  loop = 0
  status = True
  # pipeline of inference
  while cap.isOpened():
    # read an image
    ret, img = cap.read()
    if not ret:
      break
    if loop >= loops:
      break
    # preprocess
    data = preprocess(img, detected_size)
    input_data = {input_name: np.array([data], dtype=np.float32)}
    output = net.process(graph_name, input_data)
    # postprocess
    bboxes, classes, probs = postprocess(output, img, detected_size, threshold)
    # print result
    if compare(reference, bboxes, classes, probs, loop):
      for bbox, cls, prob in zip(bboxes, classes, probs):
        print("[tpu {}] Category: {}, Score: {:.3f}, Box: {}".format(\
            tpu_id, cls, prob, bbox))
    else:
      status = False
      break
    loop += 1
  cap.release()
  return status

if __name__ == '__main__':
  """ A YOLOv3 example.
  """
  PARSER = argparse.ArgumentParser(description='for sail det_yolov3 py test')
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
