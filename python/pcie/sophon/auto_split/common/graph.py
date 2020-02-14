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

from __future__ import print_function
import sys
if sys.version < '3':
  import Queue
else:
  import queue as Queue
sys.setrecursionlimit(1000000)

class Operator(object):
  """ A class represents ops for creating Graph.

  Attributes:
    name: A string, name of this operator.
    is_support: A boolean, if this operator supported on sophon.
    is_compute: A boolean, if this operator has large amounts of computations.
    is_danger: A boolean, False if this operator can be output of a graph.
    is_input: A boolean, if this operator is input of original model.
    is_output: A boolean, if this operator is output of original model.
    input_ops: A list of string, input operators' names of this operator.
  """
  def __init__(self, name, \
      is_support, is_compute, is_danger, is_input, is_output, input_ops):
    self.name = name
    self.is_support = is_support
    self.is_compute = is_compute
    self.is_danger = is_danger
    self.is_input = is_input
    self.is_output = is_output
    self.input_ops = set()
    for i in input_ops:
      self.input_ops.add(i)


class Subgraph(object):
  """ The subgraph splitted from a Graph.

  Attributes:
    name: A string, name of this subgraph.
    ops: A set of string, contains names of ops in this subgraph.
    input_ops: A set of string, contains names of ops,
                                which are inputs of original model.
    output_ops: A set of string, contains names of ops,
                                  which are outputs of original model.
    input_subgraphs: A dict, Format:
                              {
                                input_subgraph_name: set of op names
                              }
    output_subgraphs: A dict, Format:
                              {
                                output_subgraph_name: set of op names
                              }
    num_compute_ops: An integer, number of computing ops in this subgraph.
    support: 0 or 1, 1 represents this subgraph can be compiled by bm compiler.
    danger: A boolean, if there is output op in this subgraph can't be output,
                        this subgraph is danger, it must merge with it's
                        output_subgraphs.
    cantmerge: A set of string, a parameter for merging algorithm.
    visit: 0 or 1 or 2, a parameter for merging algorithm.

  """
  def __init__(self, name):
    """ Constructor.
    """
    self.name = name
    self.ops = set()
    self.input_ops = set()
    self.output_ops = set()
    self.input_subgraphs = dict()
    self.output_subgraphs = dict()

    self.num_compute_ops = 0
    self.support = 0
    self.cantmerge = set()
    self.visit = 0
    self.danger_ops = set()

  def is_subgraph_danger(self):
    """ Judge if this subgraph is dangerous.
        That is, need to merge with its outputs.
    """
    for outg in self.output_subgraphs:
      for outop in self.output_subgraphs[outg]:
        if outop in self.danger_ops:
          return True, outg
    return False, None

  def add_op(self, opname):
    """ Add operator for this subgraph
    """
    self.ops.add(opname)

  def has_op(self, opname):
    """ Judge if this subgraph has a specific operator.
    """
    return opname in self.ops

  def add_input_subgraph(self, name2, op_name2):
    """ Add an input subgraph for this subgraph
    """
    if name2 not in self.input_subgraphs:
      self.input_subgraphs[name2] = set()
    self.input_subgraphs[name2].add(op_name2)

  def add_output_subgraph(self, name2, op_name1):
    """ Add an output subgraph for this subgraph
    """
    assert op_name1 in self.ops
    if name2 not in self.output_subgraphs:
      self.output_subgraphs[name2] = set()
    self.output_subgraphs[name2].add(op_name1)

  def merge_an_output_subgraph(self, subgraph):
    """ Merge an subgraph into this subgraph
    """
    assert subgraph.name in self.output_subgraphs
    assert self.name != subgraph.name
    for operator in subgraph.ops:
      self.ops.add(operator)
    for inname in subgraph.input_subgraphs:
      if inname == self.name:
        continue
      if inname not in self.input_subgraphs:
        self.input_subgraphs[inname] = set()
      self.input_subgraphs[inname] = self.input_subgraphs[inname] | \
                                      subgraph.input_subgraphs[inname]
    if subgraph.name in self.input_subgraphs:
      self.input_subgraphs.pop(subgraph.name)
    for outname in subgraph.output_subgraphs:
      if outname == self.name:
        continue
      if outname not in self.output_subgraphs:
        self.output_subgraphs[outname] = set()
      self.output_subgraphs[outname] = self.output_subgraphs[outname] | \
                                        subgraph.output_subgraphs[outname]
    self.output_subgraphs.pop(subgraph.name)
    self.num_compute_ops = self.num_compute_ops + subgraph.num_compute_ops
    self.support = self.support * subgraph.support
    #self.danger = self.danger or subgraph.danger
    self.danger_ops = self.danger_ops | subgraph.danger_ops
    self.cantmerge = self.cantmerge | subgraph.cantmerge
    self.input_ops = self.input_ops | subgraph.input_ops
    self.output_ops = self.output_ops | subgraph.output_ops



class Graph(object):
  """ The general graph to get splitted subgraphs

  To use it, one just need call two functions.
  First, parse_from_operators, to parse a list a operators to a Graph.
  Then, get_final_subgraphs, to get
    final splitted results:
      A list of Subgraphs.

  """
  def __init__(self):
    self.subgraphs = dict()
    self.num_ops = 0
    self.num_subgraphs = 0
    self.num_compute_ops = 0

  def parse_from_operators(self, operators):
    """ Parsing operators and construct this Graph.
    """
    for operator in operators:
      self.add_new_subgraph(operator)
    for operator in operators:
      for iname in operator.input_ops:
        self.add_input_subgraph(operator.name, iname, iname)
        self.add_output_subgraph(iname, operator.name, iname)
    #for n in self.subgraphs:
    #  if len(self.subgraphs[n].output_subgraphs) == 0:
    #    self.subgraphs[n].danger = False

  def auto_merge(self):
    """ Automatically merge subgraphs.
    """
    self.auto_circle_merge()
    print('start auto split, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.auto_dangerous_merge()
    self.auto_circle_merge()
    print('[0/]dangerous merge done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.auto_simple_merge()
    print('[1/]simple merge done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.auto_circle_merge()
    print('[2/]merge circles done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.auto_simple_merge()
    print('[3/]simple merge done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.auto_complex_merge()
    print('[4/]complex merge done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.update_support_base_on_compute()
    print('[5/]clear small support graph done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.auto_complex_merge()
    print('[6/]complex merge done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')
    self.clear_cantmerge()
    self.auto_complex_merge(with_cantmerge=False)
    print('[7/]all done, remaining nodes: ' + \
            str(self.num_subgraphs) + '.')

  ###  for dangerous merge
  def __merge_dangerous_node(self, name, saw=None):
    """ Merge a dangerous node.
    """
    if saw is None:
      saw = set()
    self.subgraphs[name].danger = False
    outnames_ = self.subgraphs[name].output_subgraphs.keys()
    outnames = [n for n in outnames_]
    for o_name in outnames:
      if o_name in self.subgraphs:
        if self.subgraphs[o_name].danger and (o_name not in saw):
          saw.add(o_name)
          self.__merge_dangerous_node(o_name, saw)
    if name not in self.subgraphs:
      return
    outnames_ = self.subgraphs[name].output_subgraphs.keys()
    outnames = [n for n in outnames_]
    for o_name in outnames:
      if o_name in self.subgraphs:
        self.merge_an_output_subgraph(name, o_name)

  def auto_dangerous_merge(self):
    """ Automatically merge dangerous nodes
    """
    name_ = self.subgraphs.keys()
    name = [n for n in name_]
    for node in name:
      if node in self.subgraphs:
        is_danger, out_name = self.subgraphs[node].is_subgraph_danger()
        #if self.subgraphs[n].danger:
        #  self.__merge_dangerous_node(n)
        while is_danger:
          self.merge_an_output_subgraph(node, out_name)
          is_danger, out_name = self.subgraphs[node].is_subgraph_danger()

  ### for simple merge
  def is_auto_simple_merge_done(self, name=None):
    """ Judge if simple merge is done
    """
    if name is not None:
      for out in self.subgraphs[name].output_subgraphs:
        if self.subgraphs[name].support == self.subgraphs[out].support:
          if len(self.subgraphs[name].output_subgraphs) == 1:
            return False, out
          if len(self.subgraphs[out].input_subgraphs) == 1:
            return False, out
    else:
      for node in self.subgraphs:
        for out in self.subgraphs[node].output_subgraphs:
          if self.subgraphs[node].support == self.subgraphs[out].support:
            if len(self.subgraphs[node].output_subgraphs) == 1:
              return False, None
            if len(self.subgraphs[out].input_subgraphs) == 1:
              return False, None
    return True, None

  def auto_simple_merge(self):
    """ Do simple merge
    """
    isdone, _ = self.is_auto_simple_merge_done()
    while not isdone:
      names_ = self.subgraphs.keys()
      names = [n for n in names_]
      for name in names:
        if name in self.subgraphs:
          is_this_done, out = self.is_auto_simple_merge_done(name)
          while not is_this_done:
            self.merge_an_output_subgraph(name, out)
            is_this_done, out = self.is_auto_simple_merge_done(name)
      isdone, _ = self.is_auto_simple_merge_done()


  #for circle merge
  def dfs_find_circle(self, name):
    """ Recursively find circle in Graph.
    """
    assert self.subgraphs[name].visit == 0
    self.subgraphs[name].visit = 1
    for out in self.subgraphs[name].output_subgraphs:
      if self.subgraphs[out].visit == 1:
        return True, out
      elif self.subgraphs[out].visit == 2:
        continue
      else:
        has_circle, target = self.dfs_find_circle(out)
        if has_circle:
          return True, target
        else:
          self.subgraphs[out].visit = 2
    self.subgraphs[name].visit = 2
    return False, None

  def if_has_circle(self, name=None):
    """ Judge if this Graph has circle.
    """
    for node in self.subgraphs:
      self.subgraphs[node].visit = 0
    if name is not None:
      assert name in self.subgraphs
      has_circle, target = self.dfs_find_circle(name)
      if has_circle:
        return True, target
    else:
      for node in self.subgraphs:
        if not self.subgraphs[node].input_subgraphs:
        #if len(self.subgraphs[node].input_subgraphs) == 0:
          has_circle, target = self.dfs_find_circle(node)
          if has_circle:
            return True, target
    return False, None

  def get_the_circle(self, name):
    """ Get the circle started with node of name.
    """
    queue = Queue.Queue()
    paths = dict()
    paths[name] = name
    queue.put(name)
    while not queue.empty():
      ele = queue.get()
      for out in self.subgraphs[ele].output_subgraphs:
        if self.subgraphs[out].visit == 1:
          if out == name:
            return paths[ele]
          queue.put(out)
          paths[out] = paths[ele] + '->\n' + out
          # small circle may be in big circle
          self.subgraphs[out].visit = 0
    raise RuntimeError("haven't get circle from node: " + name + ".")

  def auto_circle_merge(self):
    """ Do circle merge
    """
    has, name = self.if_has_circle()
    while has:
      circle = self.get_the_circle(name)
      circle = circle.split('->\n')
      circle.remove(name)
      for node in circle:
        self.merge_an_output_subgraph(name, node)
      has, name = self.if_has_circle()

  # for complex merge
  def can_reach_from_other_road(self, name, target):
    """ Judge if node of name can reach to node of target,
        from other road.
        Target is an output of name.
    """
    assert name in self.subgraphs[target].input_subgraphs
    assert target in self.subgraphs[name].output_subgraphs

    for out in self.subgraphs[name].output_subgraphs:
      if out != target:
        for node in self.subgraphs:
          self.subgraphs[node].visit = 0
        self.subgraphs[target].visit = 1
        can_reach, _ = self.dfs_find_circle(out)
        if can_reach:
          assert _ == target
          return True
    return False

  def __is_auto_complex_merge_done(self, with_cantmerge=True):
    """ Judge if complex merge done
    """
    for node in self.subgraphs:
      for out in self.subgraphs[node].output_subgraphs:
        if self.subgraphs[node].support != self.subgraphs[out].support:
          continue
        if len(self.subgraphs[node].output_subgraphs) == 1 or \
            len(self.subgraphs[out].input_subgraphs) == 1:
          return False, None
        if with_cantmerge and (out in self.subgraphs[node].cantmerge):
          assert node in self.subgraphs[out].cantmerge
          continue
        if self.can_reach_from_other_road(node, out):
          self.subgraphs[node].cantmerge.add(out)
          self.subgraphs[out].cantmerge.add(node)
          continue
        return False, None
    return True, None

  def is_auto_complex_merge_done(self, name=None, with_cantmerge=True):
    """ Judge if complex merge is done.
    """
    if name is not None:
      for out in self.subgraphs[name].output_subgraphs:
        if self.subgraphs[name].support != self.subgraphs[out].support:
          continue
        if len(self.subgraphs[name].output_subgraphs) == 1 or \
            len(self.subgraphs[out].input_subgraphs) == 1:
          return False, out
        if with_cantmerge and (out in self.subgraphs[name].cantmerge):
          assert name in self.subgraphs[out].cantmerge
          continue
        if self.can_reach_from_other_road(name, out):
          self.subgraphs[name].cantmerge.add(out)
          self.subgraphs[out].cantmerge.add(name)
          continue
        return False, out
      return True, None
    return self.__is_auto_complex_merge_done(with_cantmerge=with_cantmerge)
      #for node in self.subgraphs:
      #  for out in self.subgraphs[node].output_subgraphs:
      #    if self.subgraphs[node].support != self.subgraphs[out].support:
      #      continue
      #    if len(self.subgraphs[node].output_subgraphs) == 1 or \
      #        len(self.subgraphs[out].input_subgraphs) == 1:
      #      return False, None
      #    if with_cantmerge and (out in self.subgraphs[node].cantmerge):
      #      assert node in self.subgraphs[out].cantmerge
      #      continue
      #    if self.can_reach_from_other_road(node, out):
      #      self.subgraphs[node].cantmerge.add(out)
      #      self.subgraphs[out].cantmerge.add(node)
      #      continue
      #    return False, None
    #return True, None

  def auto_complex_merge(self, with_cantmerge=True):
    """ Do complex merge
    """
    isdone, _ = self.is_auto_complex_merge_done(name=None, \
                                      with_cantmerge=with_cantmerge)
    while not isdone:
      names_ = self.subgraphs.keys()
      names = [n for n in names_]
      for node in names:
        if node in self.subgraphs:
          is_thisdone, out = self.is_auto_complex_merge_done(name=node, \
                                      with_cantmerge=with_cantmerge)
          while not is_thisdone:
            self.merge_an_output_subgraph(node, out)
            is_thisdone, out = self.is_auto_complex_merge_done(name=node, \
                                      with_cantmerge=with_cantmerge)
      isdone, _ = self.is_auto_complex_merge_done(name=None, \
                                      with_cantmerge=with_cantmerge)

  def update_support_base_on_compute(self):
    """ Make some small supported subgraphs unsupported.
    """
    for node in self.subgraphs:
      if self.subgraphs[node].support == 1:
        if self.subgraphs[node].num_compute_ops < self.num_compute_ops / 10:
          self.subgraphs[node].support = 0

  def clear_cantmerge(self):
    """ Clear cantmerge attribute in all subgraphs.
    """
    for node in self.subgraphs:
      self.subgraphs[node].cantmerge = set()

  def get_final_subgraphs(self):
    """ Get final result of merging nodes.
    """
    self.auto_merge()
    top_sort = self.get_topological_sort()
    ret = []
    for node in top_sort:
      ret.append(self.subgraphs[node])
    return ret

  def get_topological_sort(self):
    """ Get topological sort of final subgraphs.
    """
    indegrees = dict()
    top_sort = []
    for name in self.subgraphs:
      indegrees[name] = len(self.subgraphs[name].input_subgraphs)

    while indegrees:
      found = False
      tmpname = ''
      for i in indegrees:
        assert indegrees[i] >= 0
        if indegrees[i] == 0:
          found = True
          top_sort.append(i)
          for outg in self.subgraphs[i].output_subgraphs:
            indegrees[outg] = indegrees[outg] - 1
          tmpname = i
          break
      indegrees.pop(tmpname)
      assert found
    return top_sort

  def merge_an_output_subgraph(self, name1, name2):
    """ Do node merge.
    """
    self.subgraphs[name1].merge_an_output_subgraph(self.subgraphs[name2])
    for inname in self.subgraphs[name2].input_subgraphs:
      if inname == name1:
        continue
      if name1 not in self.subgraphs[inname].output_subgraphs:
        self.subgraphs[inname].output_subgraphs[name1] = set()
      self.subgraphs[inname].output_subgraphs[name1] = \
                          self.subgraphs[inname].output_subgraphs[name1] | \
                          self.subgraphs[inname].output_subgraphs[name2]
      self.subgraphs[inname].output_subgraphs.pop(name2)
      if name2 in self.subgraphs[inname].cantmerge:
        self.subgraphs[inname].cantmerge.remove(name2)
        self.subgraphs[inname].cantmerge.add(name1)
    for outname in self.subgraphs[name2].output_subgraphs:
      if outname == name1:
        continue
      if name1 not in self.subgraphs[outname].input_subgraphs:
        self.subgraphs[outname].input_subgraphs[name1] = set()
      self.subgraphs[outname].input_subgraphs[name1] = \
                          self.subgraphs[outname].input_subgraphs[name1] | \
                          self.subgraphs[outname].input_subgraphs[name2]
      self.subgraphs[outname].input_subgraphs.pop(name2)
      if name2 in self.subgraphs[outname].cantmerge:
        self.subgraphs[outname].cantmerge.remove(name2)
        self.subgraphs[outname].cantmerge.add(name1)

    self.subgraphs.pop(name2)
    self.num_subgraphs = self.num_subgraphs - 1
    sys.stdout.write(' After merge: ' + str(self.num_subgraphs) + '.\r')
    sys.stdout.flush()

  def add_new_subgraph(self, operator):
    """ Add new subgraph.
        Using this just at Graph construction.
    """
    assert operator.name not in self.subgraphs
    subgraph = Subgraph(operator.name)
    subgraph.add_op(operator.name)
    if operator.is_support:
      subgraph.support = 1
      if operator.is_compute:
        subgraph.num_compute_ops = 1
        self.num_compute_ops = self.num_compute_ops + 1
    if operator.is_danger:
      subgraph.danger_ops.add(operator.name)
    # s.danger = op.is_danger
    self.num_ops = self.num_ops + 1
    self.num_subgraphs = self.num_subgraphs + 1
    if operator.is_input:
      subgraph.input_ops.add(operator.name)
    if operator.is_output:
      subgraph.output_ops.add(operator.name)
    self.subgraphs[operator.name] = subgraph

  def add_input_subgraph(self, name1, name2, op_name2):
    """ Add input subgraph.
    """
    self.subgraphs[name1].add_input_subgraph(name2, op_name2)

  def add_output_subgraph(self, name1, name2, op_name1):
    """ Add output subgraph
    """
    self.subgraphs[name1].add_output_subgraph(name2, op_name1)
