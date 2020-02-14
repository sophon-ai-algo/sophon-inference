""" Holds some useful functions
"""
import re
import subprocess
import os
import zipfile


def parse_string_to_int_tuples(tuples):
  """ parse a string to tuples, while the string is:
      (x, x, x, x), (x, x)...
  """
  tuple_list = re.findall('[(](.*?)[)]', tuples)
  ret = []
  for i in tuple_list:
    int_shape = []
    shape = i.strip().split(',')
    for j in shape:
      if j:
        int_shape.append(int(j))
    ret.append(tuple(int_shape))
  return ret
      
def parse_string_to_float_tuples(tuples):
  """ parse a string to tuples, while the string is:
      (x, x, x, x), (x, x)...
  """
  tuple_list = re.findall('[(](.*?)[)]', tuples)
  ret = []
  for i in tuple_list:
    int_shape = []
    shape = i.strip().split(',')
    for j in shape:
      if j:
        int_shape.append(float(j))
    ret.append(tuple(int_shape))
  return ret

def press_to_zip(folder):
  """ press folder to folder.zip
  """
  folder = os.path.abspath(folder)
  center = folder.rfind('/')
  head = folder[:center]
  tail = folder[(center+1):]
  ret = subprocess.call('zip -r {0}.zip {0}'.format(tail), cwd=head, close_fds=True, shell=True)
  if ret != 0:
    raise RuntimeError('zip failed.')

def read_graph_infos_from_zip(model_path):
  model_path = os.path.abspath(model_path)
  model_name = model_path[(model_path.rfind('/') + 1):model_path.rfind('.')]
  z = zipfile.ZipFile(model_path)
  content = z.read("{}/graph_infos.json".format(model_name)).decode('utf-8')
  z.close()
  return content

def read_binary_from_zip_TF(model_path, idx, device):
  """ device: cpu or tpu
      for tf splitted zip
  """
  model_path = os.path.abspath(model_path)
  model_name = model_path[(model_path.rfind('/') + 1):model_path.rfind('.')]
  z = zipfile.ZipFile(model_path)
  if device == 'cpu':
    content = z.read("{0}/graph_{1}.pb".format(model_name, idx))
  else:
    assert(device == 'tpu')
    content = z.read("{0}/graph_ir_{1}/compilation.bmodel".format(model_name, idx))
  z.close()
  return content
