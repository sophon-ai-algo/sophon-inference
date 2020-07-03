'''
Setup file for sophon
'''
from __future__ import print_function
import os
import shutil
from distutils.core import setup, Extension
from setuptools import find_packages

# sophon_x86 python module
PACKAGES_X86 = ['sophon', 'sophon.auto_split', 'sophon.auto_split.common',
                'sophon.auto_split.compiler', 'sophon.auto_split.external',
                'sophon.auto_split.splitter']

# wrap sophon python module
setup(name='sophon_x86',
      version='2.1.0',
      description='Inference samples for deep learning on Sophon products.',
      author='Sophon algorithm team',
      author_email='hong.liu@bitmain.com',
      url='https://git.bitmain.vip/ai-algorithm/sophon-inference',
      long_description='''
Guide to deploying deep-learning inference networks and deep vision primitives on Sophon TPU.
''',
      packages=PACKAGES_X86,
      include_package_data=False)
