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

import os
import bmnetm
from ..common.base_compiler import Compiler


class MxnetCompiler(Compiler):
  """ Compile Mxnet models into bmodels.
  """
  def generate_compiling_script(self, compile_info):
    pass

  def check_init(self):
    """ Check initialized attributes.
    """
    pass

  def compile_model_using_bmcompiler(self, compile_info):
    model_name = compile_info["model_info"]["json"].split("-")[0]
    json_path = os.path.join(self.folder, \
                             compile_info["model_info"]["json"])
    params_path = os.path.join(self.folder, \
                               compile_info["model_info"]["params"])
    context_dir = os.path.join(self.folder, \
                               compile_info['context_dir'])
    input_names = compile_info['input_names']
    input_shapes = compile_info['input_shapes']
    bmnetm.compile(json_path, params_path, context_dir, \
                   self.target, input_shapes, model_name, \
                   input_names=input_names, opt=self.optimize, \
                   dyn=self.dynamic, cmp=self.compare)
    to_remove = os.path.join(os.getcwd(), 'bm_multi_engine_stas_0.dat')
    if os.path.isfile(to_remove):
      os.remove(to_remove)
