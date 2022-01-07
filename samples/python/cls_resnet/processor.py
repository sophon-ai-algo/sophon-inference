""" Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.

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
import json
import numpy as np

def preprocess(input_path):
  """ Preprocessing of an image.
  Args:
    input_path: Path to input image
    model_name: Name of the model

  Returns:
    A 4-dim ndarray of preprocessed image.
  """
  params = [["resize", [224, 224]],
            ["submean", [103.94, 116.78, 123.68]],
            ["transpose", [2, 0, 1]]]
  image = cv2.imread(input_path)
  for param in params:
    transform_proc = param[0]
    transform_param = param[1]
    # resize
    if transform_proc == 'resize':
      image = cv2.resize(
          image, (transform_param[1], transform_param[0]),
          interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # sub mean
    if transform_proc == 'submean':
      if isinstance(transform_param, list):
        image[:, :, 0] -= float(transform_param[0])
        image[:, :, 1] -= float(transform_param[1])
        image[:, :, 2] -= float(transform_param[2])
    # h w c -> c h w
    if transform_proc == 'transpose':
      trans_channel = transform_param
      image = np.transpose(
          image,
          (trans_channel[0], trans_channel[1], trans_channel[2])).copy()
  # return a 4-dim array
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  return image

def postprocess(out):
  """ Postprocessing of an image classification.

  Args:
    out: Output of an image classification model.

  Returns:
    Top 1 and Top 5 classification result.
  """
  if out.ndim > 2:
    out = out.reshape((out.shape[0], out.shape[1]))
  sort_idx = np.argsort(-out)
  top1_idx = sort_idx[:, 0].reshape((-1, 1)).tolist()
  top5_idx = sort_idx[:, :5].tolist()
  top1_score = []
  top5_score = []

  for i, _ in enumerate(top1_idx):
    temp1_score = []
    temp5_score = []
    for idx in top1_idx[i]:
      temp1_score.append(out[i][idx])
    top1_score.append(temp1_score)
    for idx in top5_idx[i]:
      temp5_score.append(out[i][idx])
    top5_score.append(temp5_score)

  return {"top1_idx": top1_idx, "top1_score": top1_score}, \
      {"top5_idx": top5_idx, "top5_score": top5_score}

def get_reference(compare_path):
  """ Get correct result from given file.

  Args:
    compare_path: Path to correct result file

  Returns:
    A dict contains correct result.
  """
  if compare_path:
    with open(compare_path, 'r') as f:
      compare_data = json.load(f)
      return compare_data
  return None

def compare(reference, result, dtype):
  """ Compare result.

  Args:
    reference: Correct result
    result: Output result
    dtype: Data type of model

  Returns:
    True for success and False for failure.
  """
  if not reference:
    print("No verify_files file or verify_files err.")
    return True
  ref = reference[dtype]
  ret = True
  if result != ref:
    ret = False
  if not ret:
    print("Result compare failed!")
    print("Expected result: {}".format(ref))
  return ret
