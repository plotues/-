import cv2
import os
import numpy as np
import struct
from sklearn.cluster import KMeans
from  sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from skimage import feature as ft
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.manifold import TSNE
from sklearn.metrics import fowlkes_mallows_score
import shutil

def load_mnist(path):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    return labels

def get_file_name(path):
        file_name = os.listdir(path)
        path_filenames = []
        file_name_list = []
        for file in file_name:
            if not file.startswith('.'):
                path_filenames.append(os.path.join(path, file))
                file_name_list.append(file)
        return path_filenames

def hog_precessing(img, cell_height, cell_width):
        #得到图片的HOG特征值
        height_init, width_init = np.shape(img)
        height = int(height_init/cell_height)
        width = int(width_init/cell_width)
        features = ft.hog(
                    img,
                    orientations=6,
                    pixels_per_cell=(height,width),
                    cells_per_block=(2 , 2),
                    block_norm="L1",
                    transform_sqrt=False,
                    feature_vector=True,
                    visualize=False)
        return features

def get_data_sets(path):
            path_filename = get_file_name(path)
            data = []
            for file in path_filename:
                    #遍历训练文件夹中所有图片,不用灰度读入会出错
                    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                    #标准化数据
                    hog_data = scale(hog_precessing(img, cell_height=2, cell_width=2))
                    data.append(hog_data)
            data_set = np.array(data)#把数据集转换成ndarray形式

            return data_set


def get_data_origin(path):
    path_filename = get_file_name(path)
    data = []
    for file in path_filename:
        # 遍历训练文件夹中所有图片,不用灰度读入会出错
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # 标准化数据
        data1 = scale(img)
        data.append(data1)
    data_set = np.array(data1)  # 把数据集转换成ndarray形式

    return data_set


def photo_classification(path, index, parameter):
    photo_path = os.path.join(path, '%s.png'%index)
    try:
        os.mkdir(r'D:\clustering\%s'%parameter)
        shutil.copyfile(photo_path,
                        r'D:\clustering\%s\%s.png' % (parameter, index))
    except FileExistsError:
        shutil.copyfile(photo_path,
                        r'D:\clustering\%s\%s.png' % (parameter, index))
    return None


model = jl.load(r'D:\clustering\model\model6.pkl')
data_set = get_data_sets(r'D:\clustering\testing data(100)')

reduced_data = PCA(n_components=2).fit_transform(data_set)


# data = model.fit(data_set)
# data = get_data_origin(r'D:\clustering\testing data')
# print(data_set)


# print(reduced_data)
#
# # print(reduced_data)
# # print(kmeans.fit(reduced_data))
# Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = model.predict(reduced_data)
labels = load_mnist(r'C:\Users\DELL\Desktop\testing data\t10k-labels.idx1-ubyte')
labels = labels[0:100]
score = fowlkes_mallows_score(labels,Z)
print(score)
i = 0
for i in range(Z.shape[0]):
    photo_classification(r'D:\clustering\testing data(100)', i, Z[i])


# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = model.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()