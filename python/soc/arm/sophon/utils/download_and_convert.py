""" Download test data, or download and convert models
Usage: python download_and_convert.py googlenet
"""
from __future__ import print_function
import os
import tarfile
import argparse
import hashlib
from functools import partial
from six.moves import urllib
#import zipfile
from sophon.utils.bmodel_converter import BmodelConverterCreator

def md5sum(filename):
    """compute md5 value"""
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()

def download_data(base_url, save_dir):
  """download data"""
  file_name = 'test_data.tar.gz'
  print("data download...")
  ret = True
  try:
    url_path = os.path.join(base_url, file_name)
    opener = urllib.request.URLopener()
    opener.retrieve(url_path, file_name)
    print("{} download finished!".format(file_name))
    print("unzip {} to {}".format(file_name, save_dir))
    with tarfile.open(file_name) as tar_file:
      tar_file.extractall(save_dir)
    os.remove(file_name)
    print("{} decompress finished!".format(file_name))
  except OSError:
    ret = False
  return ret

def get_md5(base_url, model_name):
  try:
    url_path = os.path.join(base_url, 'md5', model_name)
    opener = urllib.request.URLopener()
    md5_value = str(opener.open(url_path).read(), encoding='utf-8')  
    return md5_value
  except OSError:
    print("please check if {}'s md5sum value in server".format(model_name))
    return None 

def download_model(base_url, save_dir, model_name):
  """download model"""
  if model_name == 'mtcnncxx':
    model_name = 'mtcnn'
  file_name = model_name + '.tar.gz'
  model_path = os.path.join(save_dir, model_name)
  if os.path.exists(model_path):
    server_model_md5 = get_md5(base_url, model_name)
    if server_model_md5:
      md5sum_filename = os.path.join(save_dir, model_name, 'md5sum.txt')
      cache_model_md5 = open(md5sum_filename, 'r').read()
      if cache_model_md5 == server_model_md5:
        print("{} already exists!".format(model_name))
        return True
  print("{} download...".format(file_name))
  ret = True
  try:
    url_path = os.path.join(base_url, file_name)
    opener = urllib.request.URLopener()
    opener.retrieve(url_path, file_name)
    print("{} download finished!".format(file_name))
    print("unzip {} to {}".format(file_name, model_path))
    with tarfile.open(file_name) as tar_file:
      tar_file.extractall(save_dir)
    md5sum_filename = os.path.join(save_dir, model_name, 'md5sum.txt')
    md5sum_value = md5sum(file_name)
    with open(md5sum_filename, 'w') as md5sum_file:
      md5sum_file.write(md5sum_value)
    print("{} decompress finished!".format(file_name))
  except OSError:
    print("Failed to download {}".format(file_name))
    ret = False
  os.remove(file_name)
  return ret

def convert2bmodel(model_name, model_dir):
  """convert model"""
  no_convert_list = ['']
  if model_name in no_convert_list:
    return True
  ret = True
  try:
    bm_creator=BmodelConverterCreator(model_dir)
    if not bm_creator.is_cached_bmodel(model_name):
      bm_creator.create(model_name).converter()
      bm_creator.copy_md5sum(model_name)
    print("{} convert finished!".format(model_name))
  except Exception as e:
    print("Exception: ",e)
    ret = False
  return ret

def download_and_convert(base_url, model_dir, model_name):
  """download and convert"""
  ret = download_model(base_url, model_dir, model_name)
  if ret:
    ret = convert2bmodel(model_name, model_dir)
  else:
    print("Failed to download {}".format(model_name))
    exit(-1)
  if not ret:
    print("Failed to convert {}".format(model_name))
    exit(-2)


