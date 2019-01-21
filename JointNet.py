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
        self.child = []  # reference to child
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
        self.HipLeft = Node(10, unit_num, cell=cell)
        self.HipRight = Node(13, unit_num, cell=cell)
        self.SpineMid = Node(1, unit_num, cell=cell)
        self.SpineShoulderMid = Node(16, unit_num, cell=cell)
        self.SpineShoulderLeft = Node(17, unit_num, cell=cell) # for unstandard skeleton
        self.SpineShoulderRight = Node(18, unit_num, cell=cell) # for unstandard skeleton
        self.Neck = Node(2, unit_num, cell=cell)
        self.Head = Node(3, unit_num, cell=cell)
        self.ShoulderLeft = Node(4, unit_num, cell=cell)
        self.ElbowLeft = Node(5, unit_num, cell=cell)
        self.WristLeft = Node(6, unit_num, cell=cell)
        # self.HandLeft = Node(7, unit_num, cell=cell)
        self.ShoulderRight = Node(7, unit_num, cell=cell)
        self.ElbowRight = Node(8, unit_num, cell=cell)
        self.WristRight = Node(9, unit_num, cell=cell)
        # self.HandRight = Node(11, unit_num, cell=cell)
        self.KneeLeft = Node(11, unit_num, cell=cell)
        self.AnkleLeft = Node(12, unit_num, cell=cell)
        # self.FootLeft = Node(15, unit_num, cell=cell)
        self.KneeRight = Node(14, unit_num, cell=cell)
        self.AnkleRight = Node(15, unit_num, cell=cell)
        # self.FootRight = Node(19, unit_num, cell=cell)
        self.Nodes = [self.SpineBase, self.SpineMid, self.Neck, self.Head, self.ShoulderLeft, self.ElbowLeft,
                      self.WristLeft, self.ShoulderRight, self.ElbowRight, self.WristRight, self.HipLeft, self.KneeLeft,
                      self.AnkleLeft, self.HipRight, self.KneeRight, self.AnkleRight]

        # define the tree structure
        self.SpineBase.addChild([self.SpineMid, self.HipLeft, self.HipRight])
        self.SpineBase.isRoot = True
        self.SpineMid.addChild([self.SpineShoulderMid])
        self.SpineShoulderMid.addChild([self.Neck])
        self.Neck.addChild([self.Head])
        self.Head.isLeaf = True
        self.ShoulderLeft.addChild([self.ElbowLeft])
        self.ElbowLeft.addChild([self.WristLeft])
        self.WristLeft.isLeaf = True
        # self.WristLeft.addChild([self.HandLeft])
        # self.HandLeft.isLeaf = True
        self.ShoulderRight.addChild([self.ElbowRight])
        self.ElbowRight.addChild([self.WristRight])
        self.WristRight.isLeaf = True
        # self.WristRight.addChild([self.HandRight])
        # self.HandRight.isLeaf = True
        self.HipLeft.addChild([self.KneeLeft])
        self.KneeLeft.addChild([self.AnkleLeft])
        self.AnkleLeft.isLeaf = True
        # self.AnkleLeft.addChild([self.FootLeft])
        # self.FootLeft.isLeaf = True
        self.HipRight.addChild([self.KneeRight])
        self.KneeRight.addChild([self.AnkleRight])
        self.AnkleRight.isLeaf = True
        # self.AnkleRight.addChild([self.FootRight])
        # self.FootRight.isLeaf = True
        if sktype == "standard":
            self.SpineShoulderMid.addChild([self.ShoulderLeft, self.ShoulderRight])
        else:
            self.SpineShoulderMid.addChild([self.SpineShoulderLeft, self.SpineShoulderRight])
            self.SpineShoulderLeft.addChild([self.ShoulderLeft])
            self.SpineShoulderRight.addChild([self.ShoulderRight])

class BasicRecursiveNN:
    def __init__(self, hidden_size, batch_size):
        # super(BasicRecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def forward(self, root_node, inputs, jointNum_):
        jointNum = jointNum_
        outputs = [0] * jointNum
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = tf.unstack(inputs, jointNum, 2)
        final_state = self.recursive_forward(root_node, x, outputs)
        # output, hstate = root_node.cell(inputs[root_node.label], final_state)
        # outputs.append(output)
        return outputs, final_state

    def backward(self, root_node, inputs, jointNum_):
        jointNum = jointNum_
        outputs = [0] * jointNum
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = tf.unstack(inputs, jointNum, 2)
        self.recursive_backward(root_node, x, outputs)
        return outputs

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        state_size = self.hidden_size
        local_batchsize = tf.shape(inputs[0])[0]
        if node.state == None:
            node.state = node.cell.zero_state(local_batchsize, dtype=tf.float32)
        if len(node.child) == 0:
            output, hstate = node.cell(inputs[node.label], node.state)
            outputs[node.label] = output
            return hstate
        else:
            assert len(node.child) <= 3
            child_states = []
            for idx in range(len(node.child)):
                child_state = self.recursive_forward(node.child[idx], inputs, outputs)
                child_states.append(child_state)
            if len(child_states) == 1:
                node_inputs = tf.concat([inputs[node.label], child_states[0]], 1)
                output, hstate = node.cell(node_inputs, node.state)
                outputs[node.label] = output
                return hstate
            elif len(child_states) == 2:
                node_inputs = tf.concat([inputs[node.label], child_states[0], child_states[1]], 1)
                output, hstate = node.cell(node_inputs, node.state)
                outputs[node.label] = output
                return hstate
            else:
                node_inputs = tf.concat([inputs[node.label], child_states[0], child_states[1], child_states[2]], 1)
                output, hstate = node.cell(node_inputs, node.state)
                outputs[node.label] = output

                # outputs.append(node_state)
                return hstate

    def recursive_backward(self, node, inputs, outputs):
        # transmit the parent state to children joints
        state_size = self.hidden_size
        local_batchsize = tf.shape(inputs[0])[0]
        if node.state == None:
            node.state = node.cell.zero_state(local_batchsize, dtype=tf.float32)
        if node.parent == None:
            node_input = inputs[node.label]
            output, hstate = node.cell(node_input, node.state)
            node.state = hstate
            outputs[node.label] = output
        else:
            parent_state = node.parent.state
            node_input = tf.concat([inputs[node.label], parent_state], 1)
            output, hstate = node.cell(node_input, node.state)
            node.state = hstate
            outputs[node.label] = output

        if len(node.child) > 0:
            for idx in range(len(node.child)):
                self.recursive_backward(node.child[idx], inputs, outputs)
        return


class RecurrentRecursiveNN:
    def __init__(self, hidden_size, batch_size):
        super(BasicRecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def forward(self, root_node, inputs):
        jointNum = tf.shape(inputs)[2]
        outputs = [0] * jointNum
        x = tf.unstack(inputs, jointNum, 2)
        final_state = self.recursive_forward(root_node, x, outputs)
        # output, hstate = root_node.cell(inputs[root_node.label], final_state)
        # outputs.append(output)
        # states.append(final_state)
        return outputs, final_state

    def backward(self, root_node, inputs):
        jointNum = tf.shape(inputs)[2]
        outputs = [0] * jointNum
        x = tf.unstack(inputs, jointNum, 2)
        self.recursive_backward(root_node, x, outputs)
        return outputs

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
            outputs[node.label] = output
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
                node_input = tf.concat([inputs[node.label], child_states[0]], 1)
                output, hstate = node.cell(node_input, state)
                node.state = hstate
                outputs[node.label] = output
                return hstate
            elif len(child_states) == 2:
                if node.state == None:
                    state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
                else:
                    state = node.state
                node_input = tf.concat([inputs[node.label], child_states[0], child_states[1]], 1)
                output, hstate = node.cell(node_input, state)
                node.state = hstate
                outputs[node.label] = output
                return hstate
            else:
                if node.state == None:
                    state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
                else:
                    state = node.state
                node_input = tf.concat([inputs[node.label], child_states[0], child_states[1], child_states[2]], 1)
                output, hstate = node.cell(node_input, state)
                node.state = hstate
                outputs[node.label] = output
                return hstate

    def recursive_backward(self, node, inputs, outputs):
        # transmit the parent state to children joints
        state_size = self.hidden_size
        if node.parent == None:
            if node.state == None:
                state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                state = node.state
            node_input = inputs[node.label]
            output, hstate = node.cell(node_input, state)
            node.state = hstate
            outputs[node.label] = output
        else:
            if node.state == None:
                state = node.cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                state = node.state
            parent_state = node.parent.state
            node_input = tf.concat([inputs[node.label], parent_state], 1)
            output, hstate = node.cell(node_input, state)
            node.state = hstate
            outputs[node.label] = output

        if len(node.child) > 0:
            for idx in range(len(node.child)):
                self.recursive_backward(node.child[idx], inputs, outputs)
        return

class Bidirectional_tree_net:

    def __init__(self, hidden_size, batch_size):
        # super(BasicRecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def model(self, root_node, inputs, jointNum_, type='pos'):
        jointNum = jointNum_
        local_batchsize = tf.shape(inputs)[0]
        outputs = [0] * jointNum
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs = tf.unstack(inputs, jointNum, 2)

        def recursive(root_node, inputs, outputs, type='pos'):
            node = root_node
            if node.state == None:
                state = node.cell.zero_state(local_batchsize, dtype=tf.float32)
            else:
                state = node.state
            node_input = inputs[node.label]
            if type == 'pos':
                for idx in range(len(node.child)):
                    node_input = tf.concat([node_input, inputs[node.child[idx].label]], axis=1)
                if node.parent!= None:
                    node_input = tf.concat([node_input, inputs[node.parent.label]], axis=1)
            elif type == 'state':
                for idx in range(len(node.child)):
                    if node.child[idx].state == None:
                        node_input = tf.concat([node_input, node.cell.zero_state(local_batchsize, dtype=tf.float32)], axis=1)
                    else:
                        node_input = tf.concat([node_input, node.child[idx].state], axis=1)
                if node.parent != None:
                    node_input = tf.concat([node_input, node.parent.state], axis=1)
            else:
                raise ValueError('type value is not right!')
            output, hstate = node.cell(node_input, state)
            node.state = hstate
            outputs[node.label] = output
            if len(node.child) > 0:
                for idx in range(len(node.child)):
                    recursive(node.child[idx], inputs, outputs, type)
            return
        recursive(root_node, inputs, outputs, type)
        return outputs


class SequentialBiRecursiveNN:
    """
        The information flows from forward tree to backward tree or from backward tree to forward tree.
        And hidden states are shared between forward tree and backward tree.
        repeat: recurrent times

    """
    def __init__(self, hidden_size, batch_size, jointNum_):
        # super(BasicRecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.jointNum = jointNum_

    def forward(self, root_node_fw, root_node_bw, inputs, repeat=1):
        outputs_fw = [0] * self.jointNum
        outputs_bw = [0] * self.jointNum
        node_states = [None] * self.jointNum
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = tf.unstack(inputs, self.jointNum, 2)
        for i in range(repeat):
            final_state = self.recursive_forward(root_node_fw, x, outputs_fw, node_states)
            self.recursive_backward(root_node_bw, x, outputs_bw, node_states)
        return outputs_bw, node_states

    def backward(self, root_node_fw, root_node_bw, inputs):
        outputs_fw = [0] * self.jointNum
        outputs_bw = [0] * self.jointNum
        node_states = [None] * self.jointNum
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = tf.unstack(inputs, self.jointNum, 2)
        self.recursive_backward(root_node_fw, x, outputs_bw, node_states)
        final_state = self.recursive_forward(root_node_bw, x, outputs_fw, node_states)
        return outputs_fw, node_states

    def recursive_forward(self, node, inputs, outputs, node_states):
        # get states from children
        local_batchsize = tf.shape(inputs[0])[0]
        if node_states[node.label] == None:
            node_states[node.label] = node.cell.zero_state(local_batchsize, dtype=tf.float32)
        if len(node.child) == 0:
            output, hstate = node.cell(inputs[node.label], node_states[node.label])
            outputs[node.label] = output
            node_states[node.label] = hstate
            return hstate
        else:
            assert len(node.child) <= 3
            child_states = []
            for idx in range(len(node.child)):
                child_state = self.recursive_forward(node.child[idx], inputs, outputs, node_states)
                child_states.append(child_state)
            if len(child_states) == 1:
                node_inputs = tf.concat([inputs[node.label], child_states[0]], 1)
                output, hstate = node.cell(node_inputs, node_states[node.label])
                outputs[node.label] = output
                node_states[node.label] = hstate
                return hstate
            elif len(child_states) == 2:
                node_inputs = tf.concat([inputs[node.label], child_states[0], child_states[1]], 1)
                output, hstate = node.cell(node_inputs, node_states[node.label])
                outputs[node.label] = output
                node_states[node.label] = hstate
                return hstate
            else:
                node_inputs = tf.concat([inputs[node.label], child_states[0], child_states[1], child_states[2]], 1)
                output, hstate = node.cell(node_inputs, node_states[node.label])
                outputs[node.label] = output
                node_states[node.label] = hstate
                # outputs.append(node_state)
                return hstate

    def recursive_backward(self, node, inputs, outputs, node_states):
        # transmit the parent state to children joints
        local_batchsize = tf.shape(inputs[0])[0]
        if node_states[node.label] == None:
            node_states[node.label] = node.cell.zero_state(local_batchsize, dtype=tf.float32)
        if node.parent == None:
            node_input = inputs[node.label]
            output, hstate = node.cell(node_input, node_states[node.label])
            # node.state = hstate
            outputs[node.label] = output
            node_states[node.label] = hstate
        else:
            parent_state = node_states[node.parent.label]
            node_input = tf.concat([inputs[node.label], parent_state], 1)
            output, hstate = node.cell(node_input, node_states[node.label])
            # node.state = hstate
            outputs[node.label] = output
            node_states[node.label] = hstate
        if len(node.child) > 0:
            for idx in range(len(node.child)):
                self.recursive_backward(node.child[idx], inputs, outputs, node_states)
        return


def SingleRecursivemodel(inputs, batch_size, unitnum, JointNum):
    # each inputs should have a shape of [batch_size, v, joint_num]
    # v is the dimension of input features
    # the output is a collection of the tensors of all the joint nodes

    # if tf.equal(tf.convert_to_tensor(JointNum, dtype=tf.int32), tf.to_int32(tf.shape(inputs)[2])):
    #     raise ValueError("Inputs dimension and JointNum is not equal.")
    # x = tf.unstack(inputs, JointNum, 2)
    jointTree_fw = JointTree(unitnum)
    jointTree_bw = JointTree(unitnum)
    basicmodel = SequentialBiRecursiveNN(unitnum, batch_size, JointNum)
    ##### output_tensor contains the outputs and final state of the root joint
    # output_tensor_fw, _ = basicmodel.forward(jointTree_fw.SpineBase, inputs, JointNum)
    # output_tensor_bw = basicmodel.backward(jointTree_bw.SpineBase, inputs, JointNum)
    output_tensor, _ = basicmodel.forward(jointTree_fw.SpineBase, jointTree_bw.SpineBase, inputs, 2)

    weight_regression = weight_variable([unitnum, 3], 'regression_Weights')
    biases = bias_variable([3], 'regression_Biases')
    # weights = weight_variable([unitnum, n_classes])
    # biases = bias_variable([n_classes])
    outfeatures = []
    for i in range(JointNum):
        # output_tensor = tf.concat([output_tensor_fw[i], output_tensor_bw[i]], axis=1)
        regression_pos = tf.matmul(output_tensor[i], weight_regression) + biases
        outfeatures.append(regression_pos)
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
    outfeatures = tf.convert_to_tensor(output_tensor)
    outfeatures = tf.identity(outfeatures, name='outfeatures')
    outfeatures = tf.transpose(outfeatures, [1, 0, 2])  # [batch_size, Joint_num, v]
    return outfeatures

