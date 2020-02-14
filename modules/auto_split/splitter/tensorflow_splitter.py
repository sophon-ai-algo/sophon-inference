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
import tensorflow as tf
import bmnett as bm

from ..common.base_splitter import Splitter
from ..external.tensorflow_functions import get_all_ops_from_model_path
from ..external.tensorflow_functions import node_def_add_node
from ..external.tensorflow_functions import if_this_input_can_remove
from ..external.tensorflow_functions import node_def_remove_control_inputs
from ..external.tensorflow_functions import node_def_remove_inputs
from ..external.tensorflow_functions import node_def_remove_colocations
from ..external.tensorflow_functions import save_str_to_pb
from ..external.tensorflow_functions import tensorflow_infer_outputs


class TensorflowSplitter(Splitter):
  """ Split Tensorflow graph into subgraphs.
  """
  def initialize(self):
    self.platform = 'tensorflow'
    model_path = self.model_descriptor['model_path']
    ops, graph = get_all_ops_from_model_path(model_path)
    self.ops = ops
    self.graph = graph
    self.input_tensors = self.model_descriptor['input_tensors']
    self.input_names = self.input_tensors.keys()
    self.output_names = self.model_descriptor['output_names']
    self.input_ops = [i[:i.find(':')] for i in self.input_names]
    self.output_ops = [i[:i.find(':')] for i in self.output_names]


  def get_op_name(self, op):
    return op.name

  def if_op_const(self, operator):
    """ Judge if this operator is constant.
    """
    if operator.type == 'Const':
      return True
    if operator.type == 'Identity':
      return self.if_op_const(operator.inputs[0].op)
    return False

  def is_op_support(self, operator):
    unsupports = ['CropAndResize', 'StridedSlice', 'Softmax-']
    supports = ['Reshape-', 'Split-']
    if operator.type in unsupports:
      return False
    if operator.type in supports:
      return True
    op_dict = dict()
    op_dict['type'] = operator.type
    op_dict['dynamic'] = self.dynamic
    op_dict['inputs'] = list()
    op_dict_1 = dict()
    op_dict_1['type'] = operator.type
    op_dict_1['dynamic'] = self.dynamic
    op_dict_1['inputs'] = list()
    for i in operator.inputs:
      #if i.op.type == 'Const':
      if self.if_op_const(i.op):
        op_dict['inputs'].append({
            'shape': None,
            'dtype': str(i.dtype).split(' ')[1][1:-2].replace('float', 'fp'),
            'is_constant': True})
        op_dict_1['inputs'].append({
            'shape': None,
            'dtype': str(i.dtype).split(' ')[1][1:-2],
            'is_constant': True})
      else:
        op_dict['inputs'].append({
            'shape': None,
            'dtype': str(i.dtype).split(' ')[1][1:-2].replace('float', 'fp'),
            'is_constant': False})
        op_dict_1['inputs'].append({
            'shape': None,
            'dtype': str(i.dtype).split(' ')[1][1:-2],
            'is_constant': False})
    is_support = bm.op_is_supported(op_dict)
    is_support_1 = bm.op_is_supported(op_dict_1)
    return is_support or is_support_1

  def is_op_compute(self, op):
    compute_list = [
        'Conv2D',
        'BiasAdd',
        'Relu',
        'AvgPool',
        'DepthwiseConv2dNative',
        'FusedBatchNorm',
        'MatMul',
        'MaxPool',
        'Pad',
        'Relu6',
        'Softmax',
    ]
    if op.type in compute_list:
      return True
    return False

  def is_op_dangerous(self, op):
    dangerous_list = [
        'NextIteration',
        'LoopCond',
        'Pack',
        'Switch',
        'Shape',
        'StridedSlice',
        'Split',
        'LogicalAnd',
        'Assert--',
        'TensorArrayV3',
        'SpaceToBatchND',
        'BatchToSpaceND',
        'SpaceToBatch',
        'BatchToSpace',
        'Enter',
        'Range',
        'Slice',
        'Placeholder-',
    ]
    if op.type in dangerous_list:
      return True
    for out in op.outputs:
      #if 'float' not in str(out.dtype):
      #  return True
      try:
        out_shape = len(out.shape)
        if out_shape == 0:
          return True
      except ValueError:
        return False
    return False

  def is_input_op(self, op):
    if op.name in self.input_ops:
      return True
    return False


  def is_output_op(self, op):
    if op.name in self.output_ops:
      return True
    return False

  def get_inputs_list(self, op):
    input_list = []
    for c_input in op.control_inputs:
      input_list.append(c_input.name)
    for i in op.inputs:
      input_list.append(i.op.name)
    return input_list

  def __get_new_op_def(self, dtype, shape, name):
    """ Get generated op def
    """
    with tf.Graph().as_default():
      if self.dynamic:
        newop = tf.placeholder(dtype, shape=None, name=name).op
      else:
        newop = tf.placeholder(dtype, shape=shape, name=name).op
      return node_def_add_node(str(newop.node_def))

  def __get_removed_op_def(self, name):
    """ Get a op def that can be remove, because it is a const.
    """
    op_to_add = self.graph.get_operation_by_name(name)
    op_str = node_def_add_node(str(op_to_add.node_def))
    op_str = node_def_remove_control_inputs(op_str)
    if op_to_add.type == 'Const':
      op_str = node_def_remove_inputs(op_str)
    op_str = node_def_remove_colocations(op_str)
    return op_str

  def __get_inputs_node_def(self, subgraph, tensors):
    """ Get input node_def and new inputs
    """
    graph_str = ''
    ret_sub_inputs = set()
    for i in subgraph.input_subgraphs:
      for iop in subgraph.input_subgraphs[i]:
        operator = self.graph.get_operation_by_name(iop)
        if len(operator.outputs) > 1:
          idx = len(operator.outputs)
          for j in range(idx):
            newname = iop + '_{0}_sophon_auto'.format(j)
            newshape = tensors[newname + ':0'].shape
            ret_sub_inputs.add(newname + ':0')
            graph_str += self.__get_new_op_def(operator.outputs[j].dtype, \
                                               newshape, newname)
            #with tf.Graph().as_default():
            #  if self.dynamic:
            #    newop = tf.placeholder(operator.outputs[j].dtype, \
            #                            shape=None, name=newname).op
            #  else:
            #    newop = tf.placeholder(operator.outputs[j].dtype, \
            #                            shape=newshape, name=newname).op
            #  graph_str = graph_str + node_def_add_node(str(newop.node_def))
        elif len(operator.outputs) == 1:
          can_remove, add_list = if_this_input_can_remove(self.graph, iop)
          if can_remove:
            for addname in add_list:
              graph_str += self.__get_removed_op_def(addname)
              #op_to_add = self.graph.get_operation_by_name(addname)
              #op_str = node_def_add_node(str(op_to_add.node_def))
              #op_str = node_def_remove_control_inputs(op_str)
              #if op_to_add.type == 'Const':
              #  op_str = node_def_remove_inputs(op_str)
              #op_str = node_def_remove_colocations(op_str)
              #graph_str = graph_str + op_str
          else:
            ret_sub_inputs.add(iop + ':0')
            newshape = tensors[iop + ':0'].shape
            graph_str += self.__get_new_op_def(operator.outputs[0].dtype, \
                                               newshape, iop)
            #with tf.Graph().as_default():
            #  if self.dynamic:
            #    newop = tf.placeholder(operator.outputs[0].dtype, \
            #                            shape=None, name=iop).op
            #  else:
            #    newop = tf.placeholder(operator.outputs[0].dtype, \
            #                            shape=newshape, name=iop).op
            #  graph_str = graph_str + node_def_add_node(str(newop.node_def))
        else:
          with tf.Graph().as_default():
            newop = tf.constant(1.0, name=iop).op
            graph_str = graph_str + node_def_add_node(str(newop.node_def))
    return graph_str, ret_sub_inputs

  def __get_inner_node_def(self, subgraph):
    """ Get inner node def
    """
    graph_str = ''
    tmp_output_op_list = set()
    tmp_multi_input_list = set()
    for osg in subgraph.output_subgraphs:
      for osgop in subgraph.output_subgraphs[osg]:
        tmp_output_op_list.add(osgop)
    for i in subgraph.input_subgraphs:
      for iop in subgraph.input_subgraphs[i]:
        operator = self.graph.get_operation_by_name(iop)
        if len(operator.outputs) > 1:
          tmp_multi_input_list.add(iop)

    for o_name in subgraph.ops:
      operator = self.graph.get_operation_by_name(o_name)
      if operator.name in tmp_output_op_list and len(operator.outputs) > 1:
        with self.graph.as_default():
          for itag, op_output in enumerate(operator.outputs):
            newname = operator.name + "_{0}_sophon_auto".format(itag)
            newstr = node_def_add_node(str(tf.identity(op_output, \
                                            name=newname).op.node_def))
            graph_str = graph_str + newstr
      op_str = node_def_add_node(str(operator.node_def))
      for key in tmp_multi_input_list:
        idx_0 = op_str.find('input: "' + key + ':')
        while idx_0 > 0:
          idx_s = op_str.find('"', idx_0)
          idx_e = op_str.find('"', idx_s + 1)
          idx_i = op_str.find(':', idx_s, idx_e)
          idx = int(op_str[(idx_i + 1) : idx_e])
          new_str = key + '_{0}_sophon_auto'.format(idx)
          op_str = op_str.replace('input: "{0}:{1}"'.format(key, idx), \
                                  'input: "{0}"'.format(new_str))
          idx_0 = op_str.find('input: "' + key + ':')
        idx_0 = op_str.find('input: "' + key + '"')
        while idx_0 > 0:
          new_str = key + '_0_sophon_auto'
          op_str = op_str.replace('input: "'+key+'"', 'input: "'+new_str+'"')
          idx_0 = op_str.find('input: "' + key + '"')
      op_str = node_def_remove_colocations(op_str)
      graph_str = graph_str + op_str
    return graph_str

  def __get_outputs_names(self, subgraph):
    """ Get new outputs
    """
    ret_sub_outputs = set()
    for outg in subgraph.output_subgraphs:
      for oop in subgraph.output_subgraphs[outg]:
        operator = self.graph.get_operation_by_name(oop)
        if len(operator.outputs) > 1:
          num = len(operator.outputs)
          for in_index in range(num):
            ret_sub_outputs.add(oop + '_{0}_sophon_auto:0'.format(in_index))
        elif len(operator.outputs) == 1:
          can_remove, _ = if_this_input_can_remove(self.graph, oop)
          if not can_remove:
            ret_sub_outputs.add(oop + ':0')
        else:
          continue
    return ret_sub_outputs

  def save_subgraph(self, subgraph, save_folder, index, tensors):
    ret_model_info = dict()
    ret_model_info['model_path'] = 'graph_{0}.pb'.format(index)
    ret_sub_inputs = set()
    ret_sub_outputs = set()
    for ori_iname in subgraph.input_ops:
      ori_iop = self.graph.get_operation_by_name(ori_iname)
      assert len(ori_iop.outputs) == 1
      ret_sub_inputs.add(ori_iname + ':0')
    for ori_oname in subgraph.output_ops:
      ori_oop = self.graph.get_operation_by_name(ori_oname)
      assert len(ori_oop.outputs) == 1
      ret_sub_outputs.add(ori_oname + ':0')

    graph_str = ''
    new_inputs_str, new_inputs = self.__get_inputs_node_def(subgraph, tensors)
    graph_str += new_inputs_str
    for i in new_inputs:
      ret_sub_inputs.add(i)
    inner_str = self.__get_inner_node_def(subgraph)
    graph_str += inner_str
    new_outputs = self.__get_outputs_names(subgraph)
    for i in new_outputs:
      ret_sub_outputs.add(i)
    save_str_to_pb(graph_str, save_folder, 'graph_{0}.pb'.format(index))
    return ret_model_info, list(ret_sub_inputs), list(ret_sub_outputs)

#  def save_subgraph(self, subgraph, save_folder, index, tensors):
#    tmp_input_op_list = set()
#    tmp_output_op_list = set()
#    tmp_multi_input_list = set()
#    for isg in subgraph.input_subgraphs:
#      for isgop in subgraph.input_subgraphs[isg]:
#        tmp_input_op_list.add(isgop)
#    for osg in subgraph.output_subgraphs:
#      for osgop in subgraph.output_subgraphs[osg]:
#        tmp_output_op_list.add(osgop)
#
#    ret_model_info = dict()
#    ret_model_info['model_path'] = 'graph_{0}.pb'.format(index)
#    ret_sub_inputs = set()
#    ret_sub_outputs = set()
#
#    for ori_iname in subgraph.input_ops:
#      ori_iop = self.graph.get_operation_by_name(ori_iname)
#      assert len(ori_iop.outputs) == 1
#      ret_sub_inputs.add(ori_iname + ':0')
#    for ori_oname in subgraph.output_ops:
#      ori_oop = self.graph.get_operation_by_name(ori_oname)
#      assert len(ori_oop.outputs) == 1
#      ret_sub_outputs.add(ori_oname + ':0')
#
#    graph_str = ''
#    for i in subgraph.input_subgraphs:
#      for iop in subgraph.input_subgraphs[i]:
#        op = self.graph.get_operation_by_name(iop)
#        if len(op.outputs) > 1:
#          tmp_multi_input_list.add(iop)
#          idx = len(op.outputs)
#          for ii in range(idx):
#            newname = iop + '_{0}_sophon_auto'.format(ii)
#            newshape = tensors[newname + ':0'].shape
#            ret_sub_inputs.add(newname + ':0')
#            with tf.Graph().as_default():
#              if self.dynamic:
#                newop = tf.placeholder(op.outputs[ii].dtype, \
#                                        shape=None, name=newname).op
#              else:
#                newop = tf.placeholder(op.outputs[ii].dtype, \
#                                        shape=newshape, name=newname).op
#              graph_str = graph_str + node_def_add_node(str(newop.node_def))
#        elif len(op.outputs) == 1:
#          can_remove, add_list = if_this_input_can_remove(self.graph, iop)
#          if can_remove:
#            for addname in add_list:
#              op_to_add = self.graph.get_operation_by_name(addname)
#              op_str = node_def_add_node(str(op_to_add.node_def))
#              op_str = node_def_remove_control_inputs(op_str)
#              if op_to_add.type == 'Const':
#                op_str = node_def_remove_inputs(op_str)
#              op_str = node_def_remove_colocations(op_str)
#              graph_str = graph_str + op_str
#          else:
#            ret_sub_inputs.add(iop + ':0')
#            newshape = tensors[iop + ':0'].shape
#            with tf.Graph().as_default():
#              if self.dynamic:
#                newop = tf.placeholder(op.outputs[0].dtype, \
#                                        shape=None, name=iop).op
#              else:
#                newop = tf.placeholder(op.outputs[0].dtype, \
#                                        shape=newshape, name=iop).op
#              graph_str = graph_str + node_def_add_node(str(newop.node_def))
#        else:
#          with tf.Graph().as_default():
#            newop = tf.constant(1.0, name=iop).op
#            graph_str = graph_str + node_def_add_node(str(newop.node_def))
#
#    for o in subgraph.ops:
#      op = self.graph.get_operation_by_name(o)
#      if op.name in tmp_output_op_list and len(op.outputs) > 1:
#        with self.graph.as_default():
#          for itag, op_output in enumerate(op.outputs):
#            newname = op.name + "_{0}_sophon_auto".format(itag)
#            newstr = node_def_add_node(str(tf.identity(op_output, \
#                                            name=newname).op.node_def))
#            graph_str = graph_str + newstr
#      op_str = node_def_add_node(str(op.node_def))
#      for key in tmp_multi_input_list:
#        idx_0 = op_str.find('input: "' + key + ':')
#        while idx_0 > 0:
#          idx_s = op_str.find('"', idx_0)
#          idx_e = op_str.find('"', idx_s + 1)
#          idx_i = op_str.find(':', idx_s, idx_e)
#          idx = int(op_str[(idx_i + 1) : idx_e])
#          new_str = key + '_{0}_sophon_auto'.format(idx)
#          op_str = op_str.replace('input: "{0}:{1}"'.format(key, idx), \
#                                  'input: "{0}"'.format(new_str))
#          idx_0 = op_str.find('input: "' + key + ':')
#        idx_0 = op_str.find('input: "' + key + '"')
#        while idx_0 > 0:
#          new_str = key + '_0_sophon_auto'
#          op_str = op_str.replace('input: "'+key+'"', 'input: "'+new_str+'"')
#          idx_0 = op_str.find('input: "' + key + '"')
#      op_str = node_def_remove_colocations(op_str)
#      graph_str = graph_str + op_str
#    for og in subgraph.output_subgraphs:
#      for oop in subgraph.output_subgraphs[og]:
#        op = self.graph.get_operation_by_name(oop)
#        if len(op.outputs) > 1:
#          num = len(op.outputs)
#          for in_index in range(num):
#            ret_sub_outputs.add(oop + '_{0}_sophon_auto:0'.format(in_index))
#        elif len(op.outputs) == 1:
#          can_remove, _ = if_this_input_can_remove(self.graph, oop)
#          if not can_remove:
#            ret_sub_outputs.add(oop + ':0')
#        else:
#          continue
#    save_str_to_pb(graph_str, save_folder, 'graph_{0}.pb'.format(index))
#    return ret_model_info, list(ret_sub_inputs), list(ret_sub_outputs)

  def infer_output_tensors(self, save_folder, model_info, \
                          sub_inputs, sub_outputs, \
                          tensors):
    model_path = os.path.join(save_folder, model_info['model_path'])
    input_dict = dict()
    for s_input in sub_inputs:
      input_dict[s_input] = tensors[s_input]
    results = tensorflow_infer_outputs(model_path, input_dict, sub_outputs)
#    ret_shapes = []
#    for r in results:
#      ret_shapes.append(r.shape)
    return results

  def get_tensor_dtype(self, tensor_name):
    tensor_type = str(self.graph.get_tensor_by_name(tensor_name).dtype)
    ret_str = tensor_type[9:(len(tensor_type)-2)]
    return ret_str

  def destroy(self):
    pass
    #if hasattr(self, 'sess') and (self.sess is not None):
    #  self.sess.close()
    #  self.sess = None
