'''
Setup file for sophon
'''
from __future__ import print_function
import os
import shutil
from distutils.core import setup, Extension
from setuptools import find_packages

# check sail pylib status
# AARCH64_PATH = '../../build/lib/sail.cpython-35m-x86_64-linux-gnu.so'
AARCH64_PATH = '../../../build/lib/sail.so'
DST_PATH = './sophon'

if os.path.exists(AARCH64_PATH):
  try:
    shutil.copy(AARCH64_PATH, DST_PATH)
  except shutil.SameFileError:
    pass
  # sophon_aarch64 python module
  PACKAGES_AARCH64 = ['sophon', 'sophon.auto_runner', 'sophon.auto_runner.common',
                      'sophon.auto_runner.runner', 'sophon.auto_runner.external',
                      'sophon.utils', 'sophon.algokit',
                      'sophon.algokit.algo_cv', 'sophon.algokit.algo_cv.cls',
                      'sophon.algokit.algo_cv.det', 'sophon.algokit.algo_cv.seg',
                      'sophon.algokit.algo_nlp', 'sophon.algokit.algo_speech',
                      'sophon.algokit.algofactory',
                      'sophon.algokit.engine', 'sophon.algokit.libs',
                      'sophon.algokit.libs.extend_layer', 'sophon.algokit.utils']
  setup(name='sophon_arm',
        version='2.1.0',
        description='Inference samples for deep learning on Sophon products.',
        author='Sophon algorithm team',
        author_email='hong.liu@bitmain.com',
        url='https://git.bitmain.vip/ai-algorithm/sophon-inference',
        long_description='''
  Guide to deploying deep-learning inference networks and deep vision primitives on Sophon TPU.
  ''',
        packages=PACKAGES_AARCH64,
        include_package_data=True)
else:
  raise FileNotFoundError("sail lib not found")
