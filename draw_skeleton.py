# Plots the results from the 3D pose graph optimization.
#
#
#
# The files have the following format:
#   ID joint num x y z [ Batch Size, Joint Number ,3]


import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from optparse import OptionParser
import os
import json

def loadJasondata(filepath):
    # print(filepath)
    assert os.path.exists(filepath), (
        'Wrong data path')
    with open(filepath) as f:
        data = json.load(f)
    return data

def Norm(pose_array):
    Norm_pos = np.zeros([pose_array.shape[0], 17, 3])
    mean_pos = np.zeros([3])
    std_pos = np.zeros([3])
    mean_pos[0] = np.mean(pose_array[:, :, 0])
    mean_pos[1] = np.mean(pose_array[:, :, 1])
    mean_pos[2] = np.mean(pose_array[:, :, 2])
    std_pos[0] = np.std(pose_array[:, :, 0])
    std_pos[1] = np.std(pose_array[:, :, 1])
    std_pos[2] = np.std(pose_array[:, :, 2])
    for i in range(pose_array.shape[0]):
        for j in range(17):
            Norm_pos[i, j, 0] = (pose_array[i, j, 0] - mean_pos[0])/std_pos[0]
            Norm_pos[i, j, 1] = (pose_array[i, j, 1] - mean_pos[1])/std_pos[1]
            Norm_pos[i, j, 2] = (pose_array[i, j, 2] - mean_pos[2])/std_pos[2]
    return Norm_pos, mean_pos, std_pos

def set_axes_equal(axes):
    ''' Sets the axes of a 3D plot to have equal scale. '''
    x_limits = axes.get_xlim3d()
    y_limits = axes.get_ylim3d()
    z_limits = axes.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    length = 0.5 * max([x_range, y_range, z_range])
    axes.set_xlim3d([x_middle - length, x_middle + length])
    axes.set_ylim3d([y_middle - length, y_middle + length])
    axes.set_zlim3d([z_middle - length, z_middle + length])

def draw_skeleton(path_preds, path_gt, path_corrupted, path_label):
    draw_line = np.array([
        [3, 2, 16, 1, 0],
        [16, 4, 5, 6],
        [16, 7, 8, 9],
        [0, 10, 11, 12],
        [0, 13, 14, 15],
    ])

    batch_size = 64
    Joint_num = 17
    v = 3

    parser = OptionParser()
    parser.add_option("-e", "--axes_equal", action="store_true", dest="axes_equal",
                       default="", help="Make the plot axes equal.")
    (options, args) = parser.parse_args()

    # Read the original and optimized poses files.
    APE_pose_gt_load = np.reshape(np.asarray(loadJasondata(path_gt)), [-1, 17, 3])
    pose_corrupted = np.reshape(np.asarray(loadJasondata(path_corrupted)), [-1, v])
    # APE_train_gt, gt_train_mean, gt_train_stddev = Norm(APE_train_gt_load)
    poses_original = np.reshape(APE_pose_gt_load, [-1, v])
    poses_pred = np.reshape(np.asarray(loadJasondata(path_preds)), [-1, v])
    pose_label = np.reshape(np.asarray(loadJasondata(path_label)), [-1])

    for frame in range(100):
        idx = np.random.randint(len(pose_label))
        poses_original_dw = poses_original[idx*Joint_num:(idx+1)*Joint_num, :]
        poses_corrupted_dw = pose_corrupted[idx*Joint_num:(idx+1)*Joint_num, :]
        #print(poses_original[:,0])
        poses_pred_dw = poses_pred[idx*Joint_num:(idx+1)*Joint_num, :]

        # Plots the results for the specified poses.
        figure = plot.figure()
        ax = plot.axes(projection='3d')
        for i in range(draw_line.shape[0]):
            ax.plot3D(poses_original_dw[draw_line[i], 0], poses_original_dw[draw_line[i], 2], poses_original_dw[draw_line[i], 1], 'green')
            ax.scatter(poses_original_dw[draw_line[i], 0], poses_original_dw[draw_line[i], 2], poses_original_dw[draw_line[i], 1], s=10, c='green')

            ax.plot3D(poses_corrupted_dw[draw_line[i], 0], poses_corrupted_dw[draw_line[i], 2],
                      poses_corrupted_dw[draw_line[i], 1], 'red')
            ax.scatter(poses_corrupted_dw[draw_line[i], 0], poses_corrupted_dw[draw_line[i], 2],
                       poses_corrupted_dw[draw_line[i], 1], s=10, c='red')
            difference = [poses_original_dw[0, 0] - poses_pred_dw[0, 0] + poses_pred_dw[draw_line[i], 0],
                           poses_original_dw[0, 2] - poses_pred_dw[0, 2] + poses_pred_dw[draw_line[i], 2],
                           poses_original_dw[0, 1] - poses_pred_dw[0, 1] + poses_pred_dw[draw_line[i], 1]]
            ax.plot3D(difference[0], difference[1], difference[2], 'blue')
            ax.scatter(difference[0], difference[1], difference[2], s=10, c='blue')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            figure.suptitle(pose_label[idx])

    # for angle in range(0, 360):
    #     ax.view_init(-30, angle)
    #     plot.draw()
    #     plot.pause(.001)

    # if poses_original is not None:
    #   axes = plot.subplot(1, 2, 1, projection='3d')
    #   plot.plot(poses_original[:,0], poses_original[:,1], poses_original[:,2],
    #             '-', alpha=0.5, color="green")
    #   plot.title('Original')
    #   if options.axes_equal:
    #     axes.set_aspect('equal')
    #     set_axes_equal(axes)
    #
    # if poses_optimized is not None:
    #   axes = plot.subplot(1, 2, 2, projection='3d')
    #   plot.plot(poses_optimized[:,0], poses_optimized[:,1], poses_optimized[:,2],
    #             '-', alpha=0.5, color="blue")
    #   plot.title('Optimized')
    #   if options.axes_equal:
    #     axes.set_aspect('equal')
    #     set_axes_equal(plot.gca())

    # Show the plot
        plot.show()
    return
def main(is_training):
    path_pred = "./SeBi_models/predtrain_pos.json" if is_training==True else "./SeBi_models/pred_pos.json"
    path_gt = './APEdataset/json_file/train/APE_train_gt.json' if is_training==True else './APEdataset/json_file/test/APE_test_gt.json'
    path_label = './APEdataset/json_file/train/APE_drop_train_label.json' if is_training==True else './APEdataset/json_file/test/APE_drop_test_label.json'
    path_corrupted = './APEdataset/json_file/train/APE_train.json' if is_training==True else './APEdataset/json_file/test/APE_test.json'
    draw_skeleton(path_pred, path_gt, path_corrupted, path_label)

if __name__=='__main__':
    main(True)