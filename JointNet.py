from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

class Node:  # a node in the tree
    def __init__(self, label, unit_num, cell="GRU"):
        self.label = label
        if cell=="RNN":
            self.cell = tf.nn.rnn_cell.RNNCell(unit_num)
        elif cell=="LSTM":
            self.cell = tf.nn.rnn_cell.LSTMCell(unit_num)
        else:
            self.cell = tf.nn.rnn_cell.GRUCell(unit_num)

        self.parent = None  # reference to parent
        self.child = []  # reference to left child
        self.childnum = 0  # number of child
        self.state = None
        # true if it's a leaf joint
        self.isLeaf = False
        # true if it's a root joint
        self.isRoot = False


    def addChild(self, child_nodes):
        for _ in range(len(child_nodes)):
            child_nodes[_].parent = self
        self.child.extend(child_nodes)

class JointTree:
    """The information flows from leaf joints to root joint.

        When standard is True, the skeleton is a standard skeleton in which
        the SpineShoulder joint is not splitted to 3 joints. Otherwise, it is
        splitted for convenience of kinematics calculation.

    """

    def __init__(self, unit_num, cell="GRU", sktype="standard"):
        self.unit_num = unit_num
        self.cell = cell
        # define joint nodes
        self.SpineBase = Node(0, unit_num, cell=cell)
        self.HipLeft = Node(12, unit_num, cell=cell)
        self.HipRight = Node(16, unit_num, cell=cell)
        self.SpineMid = Node(1, unit_num, cell=cell)
        self.SpineShoulderMid = Node(20, unit_num, cell=cell)
        self.SpineShoulderLeft = Node(21, unit_num, cell=cell) # for unstandard skeleton
        self.SpineShoulderRight = Node(22, unit_num, cell=cell) # for unstandard skeleton
        self.Neck = Node(2, unit_num, cell=cell)
        self.Head = Node(3, unit_num, cell=cell)
        self.ShoulderLeft = Node(4, unit_num, cell=cell)
        self.ElbowLeft = Node(5, unit_num, cell=cell)
        self.WristLeft = Node(6, unit_num, cell=cell)
        self.HandLeft = Node(7, unit_num, cell=cell)
        self.ShoulderRight = Node(8, unit_num, cell=cell)
        self.ElbowRight = Node(9, unit_num, cell=cell)
        self.WristRight = Node(10, unit_num, cell=cell)
        self.HandRight = Node(11, unit_num, cell=cell)
        self.KneeLeft = Node(13, unit_num, cell=cell)
        self.AnkleLeft = Node(14, unit_num, cell=cell)
        self.FootLeft = Node(15, unit_num, cell=cell)
        self.KneeRight = Node(17, unit_num, cell=cell)
        self.AnkleRight = Node(18, unit_num, cell=cell)
        self.FootRight = Node(19, unit_num, cell=cell)

        # define the tree structure
        self.SpineBase.addChild([self.SpineMid, self.HipLeft, self.HipRight])
        self.SpineBase.isRoot = True
        self.SpineMid.addChild([self.SpineShoulderMid])
        self.SpineShoulderMid.addChild([self.Neck])
        self.Neck.addChild([self.Head])
        self.Head.isLeaf = True
        self.ShoulderLeft.addChild([self.ElbowLeft])
        self.ElbowLeft.addChild([self.WristLeft])
        self.WristLeft.addChild([self.HandLeft])
        self.HandLeft.isLeaf = True
        self.ShoulderRight.addChild([self.ElbowRight])
        self.ElbowRight.addChild([self.WristRight])
        self.WristRight.addChild([self.HandRight])
        self.HandRight.isLeaf = True
        self.HipLeft.addChild([self.KneeLeft])
        self.KneeLeft.addChild([self.AnkleLeft])
        self.AnkleLeft.addChild([self.FootLeft])
        self.FootLeft.isLeaf = True
        self.HipRight.addChild([self.KneeRight])
        self.KneeRight.addChild([self.AnkleRight])
        self.AnkleRight.addChild([self.FootRight])
        self.FootRight.isLeaf = True
        if sktype == "standard":
            self.SpineShoulderMid.addChild([self.ShoulderLeft, self.ShoulderRight])
        else:
            self.SpineShoulderMid.addChild([self.SpineShoulderLeft, self.SpineShoulderRight])
            self.SpineShoulderLeft.addChild([self.ShoulderLeft])
            self.SpineShoulderRight.addChild([self.ShoulderRight])

class BasicRecursiveNN:
    def __init__(self, hidden_size, batch_size):
        super(BasicRecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def forward(self, root_node, inputs):
        outputs = []
        jointNum = tf.shape(inputs)[2]
        x = tf.unstack(inputs, jointNum, 2)
        final_state = self.recursive_forward(root_node, x, outputs)
        # output, hstate = root_node.cell(inputs[root_node.label], final_state)
        # outputs.append(output)
        return outputs, final_state

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        state_size = self.hidden_size
        if len(node.child) == 0:
            state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
            output, hstate = node.cell(inputs[node.label], state)
            outputs.append(output)
            return hstate
        else:
            assert len(node.child) <= 3
            child_states = []
            for idx in range(len(node.child)):
                child_state = self.recursive_forward(node.child[idx], inputs, outputs)
                child_states.append(child_state)
            if len(child_states) == 1:
                output, hstate = node.cell(inputs[node.label], child_states[0])
                outputs.append(output)
                return hstate
            elif len(child_states) == 2:
                node_state = tf.nn.rnn_cell_impl._Linear(tf.concat([child_states[0], child_states[1]], 1), state_size)
                output, hstate = node.cell(inputs[node.label], node_state)
                outputs.append(output)
                return hstate
            else:
                node_state = tf.nn.rnn_cell_impl._Linear(tf.concat([child_states[0], child_states[1],
                                                                    child_states[3]], 1), state_size)
                output, hstate = node.cell(inputs[node.label], node_state)
                outputs.append(output)

                # outputs.append(node_state)
                return hstate


class RecurrentRecursiveNN:
    def __init__(self, hidden_size, batch_size):
        super(BasicRecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def forward(self, root_node, inputs):
        outputs = []
        jointNum = tf.shape(inputs)[2]
        x = tf.unstack(inputs, jointNum, 2)
        final_state = self.recursive_forward(root_node, x, outputs)
        # output, hstate = root_node.cell(inputs[root_node.label], final_state)
        # outputs.append(output)
        # states.append(final_state)
        return outputs, final_state

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        state_size = self.hidden_size
        if len(node.child) == 0:
            if node.state == None:
                state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                state = node.state
            output, hstate = node.cell(inputs[node.label], state)
            node.state = hstate
            outputs.append(output)
            return hstate
        else:
            assert len(node.child) <= 3
            child_states = []
            for idx in range(len(node.child)):
                child_state = self.recursive_forward(node.child[idx], inputs, outputs)
                child_states.append(child_state)
            if len(child_states) == 1:
                if node.state == None:
                    state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
                else:
                    state = node.state
                state = tf.concat([state, child_states[0]], 1)
                node_state = tf.nn.rnn_cell_impl._Linear(state, state_size)
                output, hstate = node.cell(inputs[node.label], node_state)
                node.state = hstate
                outputs.append(output)
                return hstate
            elif len(child_states) == 2:
                if node.state == None:
                    state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
                else:
                    state = node.state
                state = tf.concat([state, child_states[0], child_states[1]], 1)
                node_state = tf.nn.rnn_cell_impl._Linear(state, state_size)
                output, hstate = node.cell(inputs[node.label], node_state)
                node.state = hstate
                outputs.append(output)
                return hstate
            else:
                if node.state == None:
                    state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
                else:
                    state = node.state
                state = tf.concat([state, child_states[0], child_states[1], child_states[2]], 1)
                node_state = tf.nn.rnn_cell_impl._Linear(state, state_size)
                output, hstate = node.cell(inputs[node.label], node_state)
                node.state = hstate
                outputs.append(output)
                return hstate

def SequenceRecursivemodel_fw(inputs, is_training, batch_size, unitnum):
    # each inputs should have a shape of [batch_size, v, joint_num, time_steps]
    # v is the dimension of input features
    # the output is a collection of the tensors of all the joint nodes

    time_steps = tf.shape(inputs)[3]
    x = tf.unstack(inputs, time_steps, 3)
    jointTree = JointTree(unitnum)
    basicmodel = BasicRecursiveNN(unitnum, batch_size)
    outfeatures = []
    for i in range(time_steps):
        output_tensor = basicmodel.forward(jointTree.SpineBase, x[i])
        outfeatures.append(output_tensor)
    # fully connect across time sequence
    return outfeatures



def RecurrentRecursiveNN_fw(inputs, is_training, batch_size, unitnum):
    # inputs should have a shape of [batch_size, v, joint_num, time_steps]
    # v is the dimension of input features
    # the output is a collection of the tensors of all the joint nodes

    time_steps = tf.shape(inputs)[3]
    x = tf.unstack(inputs, time_steps, 3)
    jointTree = JointTree(unitnum)
    RRNNmodel = RecurrentRecursiveNN(unitnum, batch_size)
    outfeatures = []
    for i in range(time_steps):
        output_tensor = RRNNmodel.forward(jointTree.SpineBase, x[i])
    outfeatures = tf.identity(output_tensor, name='outfeatures')
    return outfeatures