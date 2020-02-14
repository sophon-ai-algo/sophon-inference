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
import subprocess
import sys
from ..common.base_compiler import Compiler

PY_COMAND = 'python'
if sys.version > '3':
  PY_COMAND = 'python3'


class PytorchCompiler(Compiler):
  """ Compile tf graphs into bmodels.
  """
  def check_init(self):
    assert self.platform == 'pytorch'
    assert self.layout == 'NCHW'

  def generate_compiling_script(self, compile_info):
    model = compile_info['model_info']['pth_path']
    shapes = compile_info['input_shapes']
    outdir = os.path.join(self.folder, compile_info['context_dir'])
    outdir_ = outdir.split('/')
    net_name = 'auto_pytorch_{0}_{1}'.format(outdir_[-2], outdir_[-1])
    ret = "import bmnetp as bm\n"
    ret = ret + "import os\n\n"
    ret = ret + "model='{0}'\n".format(model)
    ret = ret + "shapes={0}\n".format(shapes)
    ret = ret + "outdir='{0}'\n".format(compile_info['context_dir'])
    ret = ret + "target='{0}'\n".format(self.target)
    ret = ret + "net_name='{0}'\n\n".format(net_name)
    ret = ret + "bm.compile(model, shapes, net_name=net_name, " + \
                "outdir=outdir, target=target, " + \
                "opt=2, cmp=True, dyn={0})\n\n".format(self.dynamic)
    ret = ret + "os.remove('{0}.grp')\n\n".format(net_name)
    ret = ret + "# os.remove('bm_multi_engine_stas_0.dat')\n\n"
    with open(os.path.join(self.folder, \
        'compile_to_{0}.py'.format(compile_info['context_dir'])), \
        'w+') as save_stream:
      save_stream.write(ret)

  def compile_model_using_bmcompiler(self, compile_info):
    ret = subprocess.call([PY_COMAND, \
                    'compile_to_{0}.py'.format(compile_info['context_dir'])], \
                    cwd=self.folder, close_fds=True)
    if ret != 0:
      raise RuntimeError("compile failed: {}".format\
            ('compile_to_{0}.py'.format(compile_info['context_dir'])))
