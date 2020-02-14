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
  input_names = [i.strip() for i in input_names_]
  output_names_ = FLAGS.output_names.split(',')
  output_names = [i.strip() for i in output_names_]
  input_shapes = parse_string_to_int_tuples(FLAGS.input_shapes)
  output_shapes = parse_string_to_int_tuples(FLAGS.output_shapes)
  print(input_names)
  print(input_shapes)
  print(output_names)
  print(output_shapes)
  assert len(input_names) == len(input_shapes)
  assert len(output_names) == len(output_shapes)
#  if FLAGS.input_range is not None:
#    assert(FLAGS.input_dtype is not None)
#    input_range = parse_string_to_float_tuples(FLAGS.input_range)
#    input_dtype = FLAGS.input_dtype.strip().split(',')
#    print(FLAGS.input_range)
#    print(FLAGS.input_dtype)
  input_shapes_dict = dict(zip(input_names, input_shapes))
  output_shapes_dict = dict(zip(output_names, output_shapes))

  split('caffe', input_shapes_dict, save_dir, \
          FLAGS.proto_path, params_path=FLAGS.weight_path, \
          outputs=output_shapes_dict, dynamic=dynamic, layout='NCHW')
  print("Splitting done!\n")

  convert(save_dir, optimize=1, compare=True, target='BM1682')
  print("Compilation done!\n")

if __name__ == "__main__":
  #PARSER = argparse.ArgumentParser()
  PARSER = argparse.ArgumentParser(add_help=False)
  help_message = 'usage: python3 -m sophon.auto_split.split_caffe [-h]' +\
                 '\n --save_dir SAVE_DIR' +\
                 '\n --proto_path PROTO_PATH' +\
                 '\n --weight_path WEIGHT_PATH' +\
                 '\n --dynamic DYNAMIC' +\
                 '\n --input_names INPUT_NAMES' +\
                 '\n --input_shapes INPUT_SHAPES' +\
                 '\n --output_names OUTPUT_NAMES' +\
                 '\n --output_shapes OUTPUT_SHAPES \n'
  PARSER.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help=help_message)
  PARSER.add_argument(
      '--save_dir',
      type=str,
      required=True,
      help='Absolute path of a directory to save splitting information.'
  )
  PARSER.add_argument(
      '--proto_path',
      type=str,
      required=True,
      help='Path to prototxt file.'
  )
  PARSER.add_argument(
      '--weight_path',
      type=str,
      required=True,
      help='Path to caffemodel file.'
  )
  PARSER.add_argument(
      '--dynamic',
      type=int,
      required=True,
      help='If dynamic mode. 0 for static, 1 for dynamic.'
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
      '--output_shapes',
      type=str,
      required=True,
      help='tuples splitted by comma, (x, x, x, x), (x, x), ...'
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
