'''
Setup file for sophon
'''
from __future__ import print_function
import os
import shutil
from distutils.core import setup, Extension
from setuptools import find_packages
import glob

# check sail pylib status
LIB_DIR = '../../build_winx64/Release/'
X86_PATH = '../../build_winx64/Release/sail.cp38-win_amd.pyd'
DST_PATH = './sophon/'

# copy modules 
def copy_dir(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src,dst)
    
copy_dir('../../modules/algokit', os.path.join(DST_PATH, 'algokit'))
copy_dir('../../modules/auto_runner', os.path.join(DST_PATH, 'auto_runner'))
copy_dir('../../modules/auto_split', os.path.join(DST_PATH, 'auto_split'))
copy_dir('../../modules/utils', os.path.join(DST_PATH, 'utils'))

# copy sail module *.pyd for windows
for pydfile in glob.glob(os.path.join(DST_PATH,"*.pyd")):
    os.remove(pydfile)
    
for root,dirs,files in os.walk(LIB_DIR):
  for file in files:
    print(file)
    if file.split('.')[-1] == 'pyd':
      X86_PATH=os.path.join(root,file)

if os.path.exists(X86_PATH):
  print("copy ", X86_PATH)
  try:
    shutil.copy(X86_PATH, DST_PATH)
  except shutil.SameFileError:
    pass
else:
  raise IOError("sail python lib not found")

pyi_name = "sophon/sail.pyi"
shutil.copy("../../src/sail.pyi",pyi_name)

# sophon python module
PACKAGES = ['sophon', 'sophon.auto_split', 'sophon.auto_split.common',
            'sophon.auto_split.compiler', 'sophon.auto_split.external',
            'sophon.auto_split.splitter', 'sophon.auto_runner',
            'sophon.auto_runner.common',
            'sophon.auto_runner.external', 'sophon.auto_runner.runner',
            'sophon.algokit', 'sophon.utils',
            'sophon.algokit.algo_cv', 'sophon.algokit.algo_cv.cls',
            'sophon.algokit.algo_cv.det', 'sophon.algokit.algo_cv.seg',
            'sophon.algokit.algo_nlp', 'sophon.algokit.algo_speech',
            'sophon.algokit.algofactory',
            'sophon.algokit.engine', 'sophon.algokit.libs',
            'sophon.algokit.libs.extend_layer', 'sophon.algokit.utils']

filehandle = open("../../git_version","r");
git_version = filehandle.readline();
print(git_version);

# wrap sophon python module
setup(name='sophon',
      version=git_version,
      description='Inference samples for deep learning on Sophon products.',
      author='Sophon algorithm team',
      url='https://github.com/sophon-ai-algo/sophon-inference',
      long_description='''
Guide to deploying deep-learning inference networks and deep vision primitives on Sophon TPU.
''',
      packages=PACKAGES,
      data_files = [pyi_name],
      include_package_data=True)
