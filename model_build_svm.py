import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from  sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from skimage import feature as ft
import joblib as jl
import matplotlib.pyplot as plt
import struct
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

start = time.clock()


class Hog_kmeans():
        def __init__(self, cell_height, cell_width,  bin_size, filenames, label_path):
                self.cell_height = cell_height
                self.cell_width = cell_width
                self.bin_size = bin_size
                self.path = filenames
                self.label_path = label_path

        def load_labels(self):
                """Load MNIST data from `path`"""
                labels_path = os.path.join(self.label_path)
                with open(labels_path, 'rb') as lbpath:
                        magic, n = struct.unpack('>II',
                                                 lbpath.read(8))
                        labels = np.fromfile(lbpath,
                                             dtype=np.uint8)
                return labels

        def get_file_name(self):
                # 取得图片目录列表
                file_name = os.listdir(self.path)
                path_filenames = []
                file_name_list = []
                for file in file_name:
                        if not file.startswith('.'):
                                path_filenames.append(os.path.join(self.path, file))
                                file_name_list.append(file)
                return path_filenames

        def hog_precessing(self, img):
                #得到图片的HOG特征值
                height_init, width_init = np.shape(img)
                height = self.cell_height
                width = self.cell_width
                features = ft.hog(
                        img,
                        orientations=self.bin_size,
                        pixels_per_cell=(height,width),
                        cells_per_block=(2 , 2),
                        block_norm="L1",
                        transform_sqrt=False,
                        feature_vector=True,
                        visualize=False)
                return features

        def get_data_sets(self):
                path_filename = self.get_file_name()
                data = []
                for file in path_filename:
                        #遍历训练文件夹中所有图片,不用灰度读入会出错
                        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                        #标准化数据
                        hog_data = scale(self.hog_precessing(img))
                        data.append(hog_data)
                data_set = np.array(data)#把数据集转换成ndarray形式

                return data_set


def main():
        # 定义簇数
        # n_digits = n_digits
        # 定义滑动框尺寸和灰度图频道数量
        setting = Hog_kmeans(cell_width=2, cell_height=2, bin_size=8, filenames=r'D:\clustering\training data',
                             label_path=r'C:\Users\DELL\Desktop\training data\train-labels.idx1-ubyte')
        data_set = setting.get_data_sets()
        data_set = data_set
        # load labels
        labels = setting.load_labels()
        labels = labels[0:10000]
        pca = PCA(n_components=100).fit_transform(data_set)#数据降维处理
        SVC(kernel='rbf', class_weight='balanced')
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        result = GridSearchCV(clf, param_grid, scoring='accuracy', n_jobs=-1, cv=3)
        result.fit(pca,labels)
        # # 保存模型
        jl.dump(value=result, filename=r'D:\clustering\model\model11.pkl')
        print('model has saved!')


if __name__ == '__main__':
        main()


        end = time.clock()
        time = end - start
        print('running time is %f s'%time)
