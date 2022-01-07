'''
Setup file for sophon
'''
from __future__ import print_function
import os
import shutil
from distutils.core import setup, Extension
from setuptools import find_packages

# check sail pylib status
LIB_DIR = '../../build/lib/'
X86_PATH = '../../build/lib/sail.cpython-35m-x86_64-linux-gnu.so'
DST_PATH = './sophon'
for root,dirs,files in os.walk(LIB_DIR):
  for file in files:
    if file.split('.')[0] == 'sail':
      X86_PATH=os.path.join(root,file)

if os.path.exists(X86_PATH):
  print("x86")
  try:
    shutil.copy(X86_PATH, DST_PATH)
  except shutil.SameFileError:
    pass
else:
  raise IOError("sail python lib not found")

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
      author_email='hong.liu@bitmain.com',
      url='https://git.bitmain.vip/ai-algorithm/sophon-inference',
      long_description='''
Guide to deploying deep-learning inference networks and deep vision primitives on Sophon TPU.
''',
      packages=PACKAGES,
      include_package_data=True)
