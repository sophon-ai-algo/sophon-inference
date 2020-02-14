""" test autodeploy of tensorflow
"""

from __future__ import print_function
import os
import shutil
import argparse
import cv2
import numpy as np
from sophon.auto_split.api import split, convert
from sophon.utils.functions import parse_string_to_int_tuples
from sophon.utils.functions import parse_string_to_float_tuples
from sophon.utils.functions import press_to_zip


def main():
  """Program entry point.
  """
  dynamic = False
  if FLAGS.dynamic != 0:
    dynamic = True
  if os.path.exists(FLAGS.save_dir) and os.path.isfile(FLAGS.save_dir):
    print("{} is a file, please input a directory path.")
    exit(-1)
  save_dir = os.path.abspath(FLAGS.save_dir)
  if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
  os.makedirs(save_dir)
  input_names_ = FLAGS.input_names.split(',')
  input_names = []
  for i in input_names_:
    j = i.strip()
    if not j.endswith(":0"):
      j = j + ':0'
      #raise RuntimeError("Wrong name: {}, tensor name should end with ':0'.".format(i))
    input_names.append(j)
  output_names_ = FLAGS.output_names.split(',')
  output_names = []
  for i in output_names_:
    j = i.strip()
    if not i.endswith(":0"):
      j = j + ':0'
      #raise RuntimeError("Wrong name: {}, tensor name should end with ':0'.".format(i))
    output_names.append(j)
  input_shapes = parse_string_to_int_tuples(FLAGS.input_shapes)
  print(input_names)
  print(input_shapes)
  assert len(input_names) == len(input_shapes)
  if FLAGS.input_range is not None:
    assert(FLAGS.input_dtype is not None)
    input_range = parse_string_to_float_tuples(FLAGS.input_range)
    input_dtype = FLAGS.input_dtype.strip().split(',')
    print(FLAGS.input_range)
    print(FLAGS.input_dtype)
  input_tensors = dict()
  for i, _ in enumerate(input_names):
    if FLAGS.input_range is None:
      input_tensors[input_names[i]] = np.ones(input_shapes[i])
      continue
    if input_dtype[i].strip() == "float":
      tmp_range = input_range[i]
      tmp_data = np.random.random_sample(size=input_shapes[i])
      tmp_data = tmp_data * (tmp_range[1] - tmp_range[0]) + tmp_range[0]
      input_tensors[input_names[i]] = tmp_data
    elif input_dtype[i].strip() == "int":
      tmp_range = input_range[i]
      input_tensors[input_names[i]] = np.random.randint(int(tmp_range[0]), int(tmp_range[1]))
    else:
      raise RuntimeError("wrong input_dtype: {}".format(input_dtype[i].strip()))

  split('tensorflow', input_tensors, save_dir, \
          FLAGS.model_path, params_path=None, \
          outputs=output_names, dynamic=dynamic, layout='NHWC')
  print("Splitting done!\n")

  convert(save_dir, optimize=1, compare=True, target=FLAGS.tpu)
  print("Compilation done!\n")

  if FLAGS.to_zip == 1:
    press_to_zip(save_dir)
    shutil.rmtree(save_dir)
    print("Zipping done!\n")


if __name__ == "__main__":
  #PARSER = argparse.ArgumentParser()
  PARSER = argparse.ArgumentParser(add_help=False)
  help_message = 'usage: python3 -m sophon.auto_split.split_tensorflow [-h]' +\
                 '\n --save_dir SAVE_DIR' +\
                 '\n --model_path MODEL_PATH' +\
                 '\n --dynamic DYNAMIC' +\
                 '\n --input_names INPUT_NAMES' +\
                 '\n --input_shapes INPUT_SHAPES' +\
                 '\n --output_names OUTPUT_NAMES \n'
  PARSER.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help=help_message)
  PARSER.add_argument(
      '--save_dir',
      type=str,
      required=True,
      help='Absolute path of a directory to save splitting information.'
  )
  PARSER.add_argument(
      '--model_path',
      type=str,
      required=True,
      help='Path to model file.'
  )
  PARSER.add_argument(
      '--dynamic',
      type=int,
      required=True,
      help='If dynamic mode. 0 for static, 1 for dynamic.'
  )
  PARSER.add_argument(
      '--to_zip',
      type=int,
      default=0,
      required=False,
      help='If save to zipfile. 0 or 1.'
  )

  PARSER.add_argument(
      '--tpu',
      type=str,
      required=True,
      help='BM1682 or BM1684.'
  )
  PARSER.add_argument(
      '--input_names',
      type=str,
      required=True,
      help='input tensor names, splitted by comma.'
  )
  PARSER.add_argument(
      '--input_shapes',
      type=str,
      required=True,
      help='tuples splitted by comma, (x, x, x, x), (x, x), ...'
  )
  PARSER.add_argument(
      '--output_names',
      type=str,
      required=True,
      help='output tensor names, splitted by comma.'
  )
  PARSER.add_argument(
      '--input_range',
      type=str,
      default=None,
      help='tuples splitted by comma, left close right open, (x, y), (x, y), ...'
  )
  PARSER.add_argument(
      '--input_dtype',
      type=str,
      default=None,
      help='dtype strs splitted by comma, int, float, float, ...'
  )
  FLAGS, UNPARSED = PARSER.parse_known_args()
  main()
