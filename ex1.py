import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy
from scipy.misc import imread

def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0.        , 0.        , 0.        ],
                            [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                            [0.49019608, 0.41960784, 0.33333333],
                            [0.02745098, 0.        , 0.        ],
                            [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                            [0.14509804, 0.12156863, 0.12941176],
                            [0.4745098 , 0.40784314, 0.32941176],
                            [0.00784314, 0.00392157, 0.02745098],
                            [0.50588235, 0.43529412, 0.34117647],
                            [0.09411765, 0.09019608, 0.11372549],
                            [0.54509804, 0.45882353, 0.36470588],
                            [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                            [0.4745098 , 0.38039216, 0.33333333],
                            [0.65882353, 0.57647059, 0.49411765],
                            [0.08235294, 0.07843137, 0.10196078],
                            [0.06666667, 0.03529412, 0.02352941],
                            [0.08235294, 0.07843137, 0.09803922],
                            [0.0745098 , 0.07058824, 0.09411765],
                            [0.01960784, 0.01960784, 0.02745098],
                            [0.00784314, 0.00784314, 0.01568627],
                            [0.8627451 , 0.78039216, 0.69803922],
                            [0.60784314, 0.52156863, 0.42745098],
                            [0.01960784, 0.01176471, 0.02352941],
                            [0.78431373, 0.69803922, 0.60392157],
                            [0.30196078, 0.21568627, 0.1254902 ],
                            [0.30588235, 0.2627451 , 0.24705882],
                            [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None



def k_means(img_pix, k, iter_num = 10):
    # the centroids
    centroids = init_centroids(img_pix, k)
    clust_array = np.empty((k, 0)).tolist()

    for iter in range(iter_num):
        for pixel in img_pix:
            min = 20 # because the max value of distance in 3D norm^2 vector is 9
            min_index = 0
            for i in range(k):
                dist = (np.linalg.norm(pixel-centroids[i]))**2

                if (dist < min):
                    min = dist
                    min_index = i
            clust_array[min_index].append(pixel)

        for i in range(k):
            if (len(clust_array[i]) != 0):
                # Zj = (1/|Cj|)* sigma(||xi||^2)
                centroids[i] = np.sum(np.array(clust_array[i]), axis=0) / len(clust_array[i])
        #print ("iter " + str(iter) + ":\n" + str(centroids) + "\n")
    return centroids








def clust(img, centroids):
    img_size = img.shape
    new_img = np.array([[[0. for k in range(img_size[2])] for j in range(img_size[1])] for i in range(img_size[0])])
    for row in range(img_size[0]):
        for col in range(img_size[1]):
            min = 4 # because the max value of distance in 3D norm vector is 3
            for Zj in centroids:
                dist = np.linalg.norm(img[row][col]-Zj)
                if (dist < min):
                    min = dist
                    new_img[row][col] = Zj
    return new_img









def norm(path = 'dog.jpeg'):
    # data preperation (loading, normalizing, reshaping)
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return A_norm, X

def show(norm_img):
    # plot the image
    plt.imshow(norm_img)
    plt.grid(False)
    plt.show()


def main():
    A, X = norm()
    #print("A is:\n" + str(A))

    #print("\n\nX is:\n" + str(X))
    #show(A)
    c = k_means(X, 2)
    XS = clust(A, c)
    show(XS)


if __name__ == '__main__':
    main()
