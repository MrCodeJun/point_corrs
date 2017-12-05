import h5py
import numpy as np
from scipy import spatial


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data1 = f['data_p1'][:]
    data2 = f['data_p2'][:]
    label = f['label'][:]
    return data1, data2, label

def shuffle_data(data1,data2,labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data1[idx, ...], data2[idx, ...], labels[idx], idx


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def regular_cells_generate(vox_size, points_scale):
    """ 
    ????????????????
    To translate voxel coordinates i, j, k to original coordinates x, y, z:

    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]
    
    """
     
    X = [2*points_scale*(i+.5-vox_size/2)/vox_size for i in range(vox_size)]
    Y = [2*points_scale*(j+.5-vox_size/2)/vox_size for j in range(vox_size)]
    Z = [2*points_scale*(k+.5-vox_size/2)/vox_size for k in range(vox_size)]
    regular_pointsX,regular_pointsY,regular_pointsZ = np.meshgrid(X,Y,Z)
    
    regular_pointsX = np.reshape(regular_pointsX,[vox_size*vox_size*vox_size,1])
    regular_pointsY = np.reshape(regular_pointsY,[vox_size*vox_size*vox_size,1])
    regular_pointsZ = np.reshape(regular_pointsZ,[vox_size*vox_size*vox_size,1]) 
    regular_points = np.concatenate([regular_pointsX,regular_pointsY,regular_pointsZ],axis = 1)
    #regular_points = tf.expand_dims(regular_points,[0]) 
    #regular_points = tf.expand_dims(regular_points,[3])
    #regular_points = tf.tile(regular_points,[batch_size,1,1,points_num])
    return regular_points   


def points2vox_features(fix_points, batch_search_points,r):
    m, n, t= batch_search_points.shape
    fix_num = fix_points.shape[0]
    batch_feature = np.zeros([m,fix_num])
    for batch_id in range(m):
        current_points = batch_search_points[batch_id,:,:]
        current_points = np.squeeze(current_points)
        kdtree_points = spatial.KDTree(list((current_points)))
        neigbor_points = kdtree_points.query_ball_point(fix_points, r , p = np.inf,eps=0)
        f = [len(neigbor_points[i]) for i in range(neigbor_points.size)]
        batch_feature[batch_id,:]=f
    return batch_feature  













        
       
        
