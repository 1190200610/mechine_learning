import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
k_list = [4, 8, 12, 16, 20, 24]


def create_data_3(num=100):
    mean = [3, -5, 12]
    cov = [[0.01, 0, 0], [0, 1, 0], [0, 0, 1]]
    data = np.random.multivariate_normal(mean, cov, size=num).T
    return rotate_z(data, 40 * np.pi / 180)


def create_data_2(num=100):
    mean = [3, -5]
    cov = [[1, 0], [0, 0.01]]
    data = np.random.multivariate_normal(mean, cov, size=num).T
    return data


def rotate_z(X, theta=0):
    matrix = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    return np.dot(matrix, X)


def pca(data, k):
    # 将数据data从D维降到k维
    d = data.shape[0]
    mean = np.mean(data, axis=1)
    zero_data = np.zeros(data.shape)
    # 零均值化
    for i in range(d):
        zero_data[i] = data[i] - mean[i]
    covM = np.dot(zero_data, zero_data.T)
    values, vectors = np.linalg.eig(covM)
    values = np.real(values)
    vectors = np.real(vectors)
    index = np.argsort(values)
    rightVectors = vectors[:, index[:-(k + 1): -1]]
    tmp_data = np.dot(rightVectors.T, zero_data)
    pca_data = np.zeros(data.shape)
    for i in range(d):
        pca_data[i] = np.dot(rightVectors[i], tmp_data) + mean[i]

    return zero_data, mean, rightVectors, pca_data


def visualize_2(data, pca_data, mean, vectors):
    draw = plt.subplot()
    draw.scatter(data[0], data[1], facecolor='green', label='Origin Data')
    draw.scatter(pca_data[0], pca_data[1], facecolor='r', label='PCA Data')

    x = [mean[0] - 3 * vectors[0], mean[0] + 3 * vectors[0]]
    y = [mean[1] - 3 * vectors[1], mean[1] + 3 * vectors[1]]
    draw.plot(x, y, color='black', label='eigenVector1 direction', alpha=0.5)

    draw.set_title('data vs pca_data', fontsize=16)
    draw.set_xlabel('$x$', fontdict={'size': 14, 'color': 'red'})
    draw.set_ylabel('$y$', fontdict={'size': 14, 'color': 'red'})

    plt.legend()
    plt.show()


def visualize_3(data, pca_data, mean, vectors):
    draw = Axes3D(plt.figure())
    draw.scatter(data[0], data[1], data[2], facecolor='green', label='Origin Data')
    draw.scatter(pca_data[0], pca_data[1], pca_data[2], facecolor='r', label='PCA Data')
    # 画出vector直线
    x = [mean[0] - 3 * vectors[0, 0], mean[0] + 3 * vectors[0, 0]]
    y = [mean[1] - 3 * vectors[1, 0], mean[1] + 3 * vectors[1, 0]]
    z = [mean[2] - 3 * vectors[2, 0], mean[2] + 3 * vectors[2, 0]]
    draw.plot(x, y, z, color='blue', label='eigenVector1 direction', alpha=1)

    x2 = [mean[0] - 3 * vectors[0, 1], mean[0] + 3 * vectors[0, 1]]
    y2 = [mean[1] - 3 * vectors[1, 1], mean[1] + 3 * vectors[1, 1]]
    z2 = [mean[2] - 3 * vectors[2, 1], mean[2] + 3 * vectors[2, 1]]
    draw.plot(x2, y2, z2, color='purple', label='eigenVector2 direction', alpha=1)

    draw.set_title('data vs pca_data', fontsize=16)
    draw.set_xlabel('$x$', fontdict={'size': 14, 'color': 'red'})
    draw.set_ylabel('$y$', fontdict={'size': 14, 'color': 'red'})
    draw.set_zlabel('$z$', fontdict={'size': 14, 'color': 'red'})

    plt.legend()
    plt.show()


def read_pca_face(file_path):
    face_list = read_face(file_path)
    for face in face_list:
        pca_list = []
        psnr_list = []
        for k in k_list:
            zero_data, mean, vector, pca_data = pca(face, int(k))
            pca_list.append(pca_data)
            psnr_list.append(psnr(face, pca_data))
        show_faces(face, pca_list, psnr_list)


def read_face(path):
    face_list = os.listdir(path)
    f_list = []
    for file in face_list:
        file_path = os.path.join(path, file)
        pic = Image.open(file_path).convert('L')
        # pic.show()
        f_list.append(np.asarray(pic))
    return f_list


def psnr(source, target):
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)


def show_faces(face, face_list, psnr_list):
    plt.figure(figsize=(20, 12), frameon=False)
    size = np.ceil((len(k_list) + 1) / 2)
    plt.subplot(2, size, 1)
    plt.title('Real Image')
    plt.imshow(face, cmap='gray')
    plt.axis('off')
    for i in range(len(k_list)):
        plt.subplot(2, size, i + 2)
        plt.title('k = ' + str(k_list) + ', PSNR = ' + '{:.2f}'.format(psnr_list[i]))
        plt.imshow(face_list[i], cmap='gray')
        plt.axis('off')
    plt.show()


def test_3D():
    data = create_data_3()
    zero_data, mean, vector, pca_data = pca(data, 2)
    visualize_3(data, pca_data, mean, vector)


def test_2D():
    data = create_data_2()
    zero_data, mean, vector, pca_data = pca(data, 1)
    visualize_2(data, pca_data, mean, vector)


def main():
    # test_2D()
    # test_3D()
    read_pca_face("data/pca_face/data")


if __name__ == '__main__':
    main()
