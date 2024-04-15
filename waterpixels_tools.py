import skimage.morphology as morpho
import cv2
import math

import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, mark_boundaries
from time import time
from skimage.measure import label

import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, mark_boundaries
from time import time
from skimage.measure import label


def open_close_smoothing(img, sigma):
    """
    This function takes an image and a sigma value as input and
    returns the smoothed image using a morphological opening and closing.
    """

    img_test = np.asarray(img, dtype=np.uint8)  # Specifying dtype is critical here
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (math.ceil(sigma * sigma / 16), math.ceil(sigma * sigma / 16)),
    )

    opening = cv2.morphologyEx(img_test, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


def morpho_gradient(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    morpho_grad = morpho.dilation(img) - morpho.erosion(img)
    morpho_grad = morpho_grad.astype("int")
    return morpho_grad


def get_grid_points(img, step):
    """
    Get the important points of the image, the middle of each square of the grid
    """

    grid_points = []
    for i in range(0, img.shape[0], step):
        for j in range(0, img.shape[1], step):
            grid_points.append((i, j))
    return grid_points


def get_grid_points_middles(img, step):
    """
    the middle of each square of the grid
    """
    grid_points = []
    for l in range(0, img.shape[0], step):
        for c in range(0, img.shape[1], step):
            middle = (l + step // 2, c + step // 2)
            if middle[0] < img.shape[0] and middle[1] < img.shape[1]:
                grid_points.append(middle)

    return grid_points


def select_markers(img_grad, grid_points, step=20) -> np.ndarray:
    """
    in each cell of the grid, select the point with the minimum gradient

    """

    n = len(grid_points)
    markers = np.empty((n, 2))
    print(markers.shape)
    for i in range(len(grid_points)):
        # get the minimum gradient in the cell
        min_grad = 255
        point = grid_points[i]

        for l in range(point[0] - step // 2, point[0] + step // 2):
            for c in range(point[1] - step // 2, point[1] + step // 2):

                if l < 0 or l >= img_grad.shape[0] or c < 0 or c >= img_grad.shape[1]:
                    continue

                if img_grad[l][c] < min_grad:
                    min_grad = img_grad[l][c]
                    min_point = (l, c)
        markers[i, 0] = min_point[0]
        markers[i, 1] = min_point[1]
    return markers


def select_group_markers(img_grad, grid_points, step=20):
    """
    in each cell of the grid, select the group od connex points with the minimum gradient
    """
    grid = img_grad.copy()
    markers = []

    def grow_marker(l, c, dico, value):
        """given a line and column it gets all the connex component of the same value"""

        if l < 0 or l >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
            return []
        elif grid[l][c] != value or grid[l][c] == -1:
            return []
        else:
            # dico[value] = dico[value] + [(l, c)]
            grid[l][c] = -1
            return (
                [(l, c)]
                + grow_marker(l + 1, c, dico, value)
                + grow_marker(l - 1, c, dico, value)
                + grow_marker(l, c + 1, dico, value)
                + grow_marker(l, c - 1, dico, value)
            )

    for point in grid_points:
        # get the minimum gradient in the cell
        min_grad = 255

        dico = (
            dict()
        )  # a dictionnary to store values of the minimum gradient in each cell

        for l in range(point[0] - step // 2, point[0] + step // 2):
            for c in range(point[1] - step // 2, point[1] + step // 2):

                if l < 0 or l >= img_grad.shape[0] or c < 0 or c >= img_grad.shape[1]:
                    continue

                if img_grad[l][c] < min_grad:
                    min_grad = img_grad[l][c]
                    if img_grad[l][c] in dico:
                        # dico[img_grad[l][c]].append(grow_marker(l, c, dico, img_grad[l][c]))
                        dico[img_grad[l][c]] = dico[img_grad[l][c]] + grow_marker(
                            l, c, dico, img_grad[l][c]
                        )
                    else:
                        dico[img_grad[l][c]] = []
                        dico[img_grad[l][c]] = dico[img_grad[l][c]] + grow_marker(
                            l, c, dico, img_grad[l][c]
                        )
        # get the longest list of points

        max_len = 0
        for key in dico:
            if len(dico[key]) > max_len:
                marker = dico[key]
                max_len = len(dico[key])
        markers.append(marker)
    return markers


def get_nearest_center(point, centers, step=20):
    """
    Get the nearest center of a point
    point : any point of the gradient
    centers : the centers of the cells of the grid
    step (sigma) : the step of the grid

    return : the nearest center and the distance between the point and the center
    warning : this is not an eucledean distance it is normalized
    """
    min_dist = 100000
    nearest_center = None

    for center in centers:
        # dist = np.linalg.norm(np.array(point) - np.array(center))
        dist = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_center = center

    assert nearest_center is not None

    return nearest_center, (2 / step) * min_dist


def select_markers_closest_to_center(
    img_grad: np.ndarray, grid_points: list, step=20
) -> np.ndarray:
    """
    in each cell of the grid, select the point with the minimum gradient
    that is closest to the center of the cell
    """

    n = len(grid_points)
    markers = np.empty((n, 2))
    print(markers.shape)
    for i in range(len(grid_points)):
        # get the minimum gradient in the cell
        min_grad = 255
        point = grid_points[i]
        smallest_distance = np.inf

        for l in range(point[0] - step // 2, point[0] + step // 2):
            for c in range(point[1] - step // 2, point[1] + step // 2):

                if l < 0 or l >= img_grad.shape[0] or c < 0 or c >= img_grad.shape[1]:
                    continue

                if (
                    img_grad[l][c] < min_grad
                    and np.sqrt((l - point[0]) ** 2 + (c - point[1]) ** 2)
                    < smallest_distance
                ):
                    min_grad = img_grad[l][c]
                    min_point = (l, c)
                    smallest_distance = np.sqrt(
                        (l - point[0]) ** 2 + (c - point[1]) ** 2
                    )

        markers[i, 0] = min_point[0]
        markers[i, 1] = min_point[1]
    return markers


def get_labeled_pixels(img_grad, markers, step=20):
    """
    Get the labeled pixels of the image
    return : a matrix with the th size of the image added to that the nearerst center and the distance
    """

    labeled_pixels = np.zeros((img_grad.shape[0], img_grad.shape[1], 3), dtype=np.int32)
    for l in range(img_grad.shape[0]):
        for c in range(img_grad.shape[1]):

            nearest_center, dist = get_nearest_center((l, c), markers, step)
            labeled_pixels[l, c, 0] = nearest_center[0]
            labeled_pixels[l, c, 1] = nearest_center[1]
            labeled_pixels[l, c, 2] = dist

    return labeled_pixels


def get_regularized_grad_label(img_grad, pixel_labels, k=4):
    """
    returns the regularized gradient of the image according
    k : spacial regularization constant
    """
    reg_grad = np.empty_like(img_grad).astype("float")

    for l in range(img_grad.shape[0]):
        for c in range(img_grad.shape[1]):
            dist_to_center = pixel_labels[l][c][2]
            reg_grad[l][c] = img_grad[l][c] + k * dist_to_center

    mx = np.max(reg_grad)

    normalized_reg_grad = (reg_grad / mx) * 255

    return normalized_reg_grad


def create_markers_image(img_grad, points_array, step=20):
    """
    img_grad is three dimensional
    create an image with the markers from a marker np array in the shape (nb_points, 2)
    """
    markers = np.zeros_like(img_grad[:, :], dtype=np.int32)
    nb_points = points_array.shape[0]

    for i in range(nb_points):
        l = points_array[i, 0]
        c = points_array[i, 1]
        markers[int(l), int(c)] = True

    return markers


def waterpixel(img_path, smoothening=10, k=2, step=50, plot=True):
    """
    This function takes an image and a sigma value as input and
    k is the spacial regularization constant
    step is the step of the grid
    returns the waterpixeled images
    """
    time1 = time()

    # loading the image

    im = skio.imread(img_path)
    if im is None:
        print("Error loading the image.")
        return

    # apply smoothening
    im_smooth = open_close_smoothing(im, smoothening)

    # calculate the gradient
    img_grad = morpho_gradient(im)

    # testing the labeled pixels method
    # get the points in the grid
    grid_points = get_grid_points_middles(img_grad, step=step)
    # selecting the markers
    markers_points = select_markers_closest_to_center(img_grad, grid_points, step=step)

    labels = get_labeled_pixels(img_grad, markers_points, step=step)
    regularized_grad = get_regularized_grad_label(img_grad, labels, k=k)

    markers_image = create_markers_image(img_grad, markers_points, step=50)
    markers = label(markers_image)
    ws = watershed(regularized_grad, markers)

    if plot:
        plt.figure()
        plt.imshow(mark_boundaries(im, ws))
        plt.figure()
        plt.imshow(ws)
        plt.show()

    time2 = time()
    print("time for the preprocessing : ", time2 - time1)
    return mark_boundaries(im, ws)


if __name__ == "__main__":
    waterpixel("./landscape.jpg", k=3, step=30)
