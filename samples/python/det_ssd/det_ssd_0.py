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
import cv2
import json
import numpy as np
import sophon.sail as sail

class PreProcessor:
  """ Preprocessing class.
  """
  def __init__(self, scale):
    """ Constructor.
    """
    self.mean = [x * scale for x in [-123, -117, -104]]

  def process(self, input):
    """ Execution function of preprocessing.
    Args:
      input: Input image

    Returns:
      A 3-dim ndarray of preprocessed image.
    """
    tmp = cv2.resize(input, (300, 300), \
                     interpolation=cv2.INTER_NEAREST).astype(np.float32)
    tmp = tmp.transpose([2, 0, 1]) # hwc -> chw
    tmp[0, :, :] += self.mean[0]
    tmp[1, :, :] += self.mean[1]
    tmp[2, :, :] += self.mean[2]
    return tmp

class PostProcessor:
  """ Postprocessing class.
  """
  def __init__(self, threshold):
    """ Constructor.
    """
    self.threshold = threshold

  def process(self, data, img_w, img_h):
    """ Execution function of postprocessing.
    Args:
      data: Inference output
      img_w: Image width
      img_h: Imgae height

    Returns:
      Detected boxes.
    """
    data = data.reshape((data.shape[2], data.shape[3]))
    ret = []
    for proposal in data:
      if proposal[2] < self.threshold:
        continue
      ret.append([
          int(proposal[1]),           # class id
          proposal[2],                # score
          int(proposal[3] * img_w),   # x0
          int(proposal[4] * img_h),   # x1
          int(proposal[5] * img_w),   # y0
          int(proposal[6] * img_h)])  # y1
    return ret

  def get_reference(self, compare_path):
    """ Get correct result from given file.
    Args:
      compare_path: Path to correct result file

    Returns:
      Correct result.
    """
    if compare_path:
      with open(compare_path, 'r') as f:
        reference = json.load(f)
        return reference["boxes"]
    return None

  def compare(self, reference, result, loop_id):
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
    data = []
    for line in result:
      cp_line = line.copy()
      cp_line[1] = "{:.8f}".format(cp_line[1])
      data.append(cp_line)
    if len(data) != len(reference):
      message = "Expected deteted number is {}, but detected {}!"
      print(message.format(len(reference), len(data)))
      return False
    ret = True
    message = "Category: {}, Score: {}, Box: [{}, {}, {}, {}]"
    fail_info = "Compare failed! Expect: " + message
    ret_info = "Result Box: " + message
    for i in range(len(data)):
      box = data[i]
      ref = reference[i]
      if box != ref:
        print(fail_info.format(ref[0], float(ref[1]), ref[2],\
                               ref[3], ref[4], ref[5]))
        print(ret_info.format(box[0], float(box[1]), box[2],\
                               box[3], box[4], box[5]))
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
  # init Engine and load bmodel
  engine = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
  # get model info
  # only one model loaded for this engine
  # only one input tensor and only one output tensor in this graph
  graph_name = engine.get_graph_names()[0]
  input_name  = engine.get_input_names(graph_name)[0]
  output_name = engine.get_output_names(graph_name)[0]
  input_dtype = engine.get_input_dtype(graph_name, input_name)
  is_fp32 = (input_dtype == sail.Dtype.BM_FLOAT32)
  scale = engine.get_input_scale(graph_name, input_name)
  # init preprocessor and postprocessor
  preprocessor = PreProcessor(scale)
  threshold = 0.59 if is_fp32 else 0.52
  postprocessor = PostProcessor(threshold)
  reference = postprocessor.get_reference(compare_path)
  cap = cv2.VideoCapture(input_path)
  status = True
  # pipeline of inference
  for i in range(loops):
    # read an image from a image file or a video file
    ret, img0 = cap.read()
    if not ret:
      break
    h, w, _ = img0.shape
    # preprocess
    data = preprocessor.process(img0)
    # inference
    input_tensors = {input_name: np.array([data], dtype=np.float32)}
    output = engine.process(graph_name, input_tensors)
    # postprocess
    dets = postprocessor.process(output[output_name], w, h)
    # print result
    if postprocessor.compare(reference, dets, i):
      for (class_id, score, x0, y0, x1, y1) in dets:
        message = '[Frame{}] Category: {}, Score: {:.3f}, Box: [{}, {}, {}, {}]'
        print(message.format(i + 1, class_id, score, x0, y0, x1, y1))
        cv2.rectangle(img0, (x0, y0), (x1, y1), (255, 0, 0), 3)
      cv2.imwrite('result-{}.jpg'.format(i + 1), img0)
    else:
      status = False
      break
  cap.release()
  return status

if __name__ == '__main__':
  """ A SSD example using opencv to decode and preprocess.
  """
  desc='decode (opencv) + preprocess (opencv) + inference (sophon inference)'
  PARSER = argparse.ArgumentParser(description=desc)
  PARSER.add_argument('--bmodel', default='', required=True)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--loops', default=1, type=int, required=False)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  PARSER.add_argument('--compare', default='', required=False)
  ARGS = PARSER.parse_args()
  status = inference(ARGS.bmodel, ARGS.input, \
                     ARGS.loops, ARGS.tpu_id, ARGS.compare)
  sys.exit(0 if status else -1)
