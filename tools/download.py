""" Download test data, or download and convert models
Usage: python download_and_convert.py googlenet
"""
from __future__ import print_function
import os
import time
import argparse
import tarfile
import sophon.utils.version as ve
from six.moves import urllib

def download_and_extract(base_url, save_dir, name, extract=True):
  """download data and extract"""
  download_path = os.path.join(save_dir, name)
  if os.path.exists(download_path):
    print("File already existed: {}, need not to download!".format(name))
    return True
  ori_name = name
  if extract:
    name = name + '.tgz'
  download_path = os.path.join(save_dir, name)
  ret = True
  try:
    url_path = os.path.join(base_url, name)
    opener = urllib.request.URLopener()
    opener.retrieve(url_path, download_path)
    print("Downloaded {}!".format(name))
    if extract:
      with tarfile.open(download_path) as tar_file:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_file, save_dir)
      os.remove(download_path)
      print("Decompressed {}!".format(name))
  except OSError:
    print("Failed to download {}!".format(name))
    ret = False
  return ret

def download_model_with_version(base_url, save_dir, name, version):
  """download bmodel and umodel"""
  ret = True
  prefix = name.split(".")[0]
  postfix = name.split(".")[1]
  model_name = prefix + '_' + version + "." + postfix
  file_name = model_name
  extract = True
  if "tgz" == postfix:
    extract = False
  ret = download_and_extract(base_url, save_dir, model_name, extract)
  # try again if download failed
  if ret == False:
    time.sleep(5)
    ret = download_and_extract(base_url, save_dir, model_name, extract)
  return ret

def main():
  """Program entry point.
  """
  base_url = 'https://sophon-file.sophon.cn/sophon-prod-s3/model/19/12/05/'
  #base_url = 'https://sophon-file.bitmain.com.cn/sophon-prod-s3/model/19/12/05/'
  #base_url = 'http://10.30.34.184:8080/sophon_model/version_test'
  save_dir = FLAGS.save_path
  version = FLAGS.version
  model_dir = save_dir
  data_dir = save_dir
  model_list = [
      'resnet50_caffe.tgz',
      'resnet50_fp32.bmodel',
      'resnet50_int8.bmodel',
      'ssd_fp32.bmodel',
      'ssd_int8.bmodel',
      'ssd_vgg_caffe.tgz',
      'yolov3_caffe.tgz',
      'yolov3_fp32.bmodel',
      'yolov3_int8.bmodel',
      'mtcnn_caffe.tgz',
      'mtcnn_fp32.bmodel',
  ]
  data_list = [
      'cls.jpg',
      'det.h264',
      'det.jpg',
      'face.jpg',
  ]
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if not version in ve.version_list:
    print("version error")
    return False
  if FLAGS.arg == 'all':
    for model_name in model_list:
      download_model_with_version(base_url, model_dir, model_name, version)
      time.sleep(1)
    for data_name in data_list:
      ret = download_and_extract(base_url, save_dir, data_name)
      time.sleep(1)
  elif FLAGS.arg == 'model_list':
    print("default model list:")
    for name in model_list:
      print("- {}".format(name))
  elif FLAGS.arg == 'test_data':
    for data_name in data_list:
      ret = download_and_extract(base_url, save_dir, data_name)
      time.sleep(1)
  elif FLAGS.arg in model_list:
    model_name = FLAGS.arg
    download_model_with_version(base_url, model_dir, model_name, version)
  elif FLAGS.arg in data_list:
    data_name = FLAGS.arg
    ret = download_and_extract(base_url, save_dir, data_name)
  else:
    raise ValueError('Invalid argument', FLAGS.arg)

if __name__ == '__main__':
  default_save_dir = os.getcwd()
  PARSER = argparse.ArgumentParser(\
    description='Download test data, or download and convert models.')
  PARSER.add_argument(
    "arg",
    type=str,
    help="Options: model_name, 'model_list', 'test_data', 'all'")
  PARSER.add_argument(
    "--save_path",
    type=str,
    default=default_save_dir,
    help="Save sophon test model and test data.")
  PARSER.add_argument(
    "--version",
    type=str,
    default=ve.__version__,
    help="Default version is latest.")
  FLAGS, UNPARSED = PARSER.parse_known_args()
  main()
