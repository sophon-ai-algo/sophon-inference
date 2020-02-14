# Copyright 2016-2022 Bitmain Technologies Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The function of config file load
"""
import os
import json
import yaml
from ..kitconfig import ChipMode


def read_config(file_path):
  """ Load the json file

  Args:
      file_path: input json file path

  Returns:
      config dict
  """
  with open(file_path, 'r') as f_config:
    json_object = json.load(f_config)
  return json_object


def dump_json(file_path, data):
  """ Dump the json file
  """
  with open(file_path, 'w') as f_config:
    json.dump(data, f_config)
  return


def load_yaml(yaml_path):
  """Load the yaml|json file

  Returns:
      A dict which contains the config info
  """
  with open(yaml_path, 'r') as f_config:
    config_dict = yaml.full_load(f_config)
  return config_dict


def load_config(config_path, model_dir, mode):
  """Extract config info from input json file
    """
  whole_config = load_yaml(config_path)
  # whole_config = read_config(config_path)
  arch = whole_config.get('arch', None)
  if arch is None:
    # one more net
    for key in whole_config:
      if 'xform' not in key:
        # assert ('arch' in whole_config[key])
        whole_config[key]['arch'] = process_arch(whole_config[key]['arch'],
                                                 model_dir, mode)
        whole_config[key]['mode'] = mode
  else:
    whole_config['arch'] = process_arch(whole_config['arch'], model_dir, mode)
    whole_config['mode'] = mode
  return whole_config


def process_arch(config, model_dir, mode):
  """Parsing configuration file
    """
  if mode in [ChipMode.BM1682, ChipMode.BM1684]:
    # if mode in [ChipMode.BM1682]:
    #     dir_name = 'bm1682model'
    # else:
    #     dir_name = 'bm1684model'
    # model_dir = os.path.join(model_dir, dir_name)
    for key in config:
      if key == 'context_path':
        config[key] = os.path.join(model_dir, config[key])
      if key == 'input_shapes':
        config[key] = [tuple(config[key][i]) for i in range(len(config[key]))]
      # other key: dynamic tpu
    return config
  else:
    raise ValueError('not valid mode: {}'.format(mode))


def load_predefined_config(model_dir, mode, arch):
  """Extract config info from predefined json file
    """
  localpath = os.path.dirname(os.path.realpath(__file__))
  if not isinstance(mode, ChipMode):
    raise ValueError('not valid mode: {}'.format(mode))
  if mode in [ChipMode.BM1682]:
    chip_name = 'bm1682'
  elif mode in [ChipMode.BM1684]:
    chip_name = 'bm1684'
  else:
    raise ValueError('not valid mode: {}'.format(mode))
  config_path = os.path.join(
      localpath, '../engine/engineconfig/{}.json'.format(arch))
  return load_config(config_path, model_dir, mode)


def load_autodeploy_config(model_dir, mode, arch, config_full_path=None):
  """Extract autodeploy config info from predefined json file
    """
  localpath = os.path.dirname(os.path.realpath(__file__))
  if not isinstance(mode, ChipMode):
    raise ValueError('not valid mode: {}'.format(mode))
  if mode in [ChipMode.BM1682]:
    chip_name = 'bm1682'
    # dir_name = 'bm1682model'
    target = 'BM1682'
  elif mode in [ChipMode.BM1684]:
    chip_name = 'bm1684'
    # dir_name = 'bm1684model'
    target = 'BM1684'
  else:
    raise ValueError('not valid mode: {}'.format(mode))
  if config_full_path is None:
    config_path = os.path.join(
        localpath,
        '../engine/autodeployconfig/{}.json'.format(arch))
  else:
    config_path = config_full_path
  config_dict = load_yaml(config_path)
  for key in config_dict:
    if key in [
        'source_path', 'subgraph_path', 'tfmodel_path', 'json_path',
        'params_path'
    ]:
      # config_dict[key] = os.path.join(model_dir, dir_name, config_dict[key])
      config_dict[key] = os.path.join(model_dir, config_dict[key])
  config_dict['target'] = target
  return config_dict
