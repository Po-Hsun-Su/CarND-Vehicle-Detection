# %%
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import cv2
import glob
# %% feature extraction
def HSL_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hhist = np.histogram(img[:,:,0], bins=32, range=(0, 180))
    shist = np.histogram(img[:,:,1], bins=32, range=(0, 256))
    vhist = np.histogram(img[:,:,2], bins=32, range=(0, 256))
    # Generating bin centers
    # bin_edges = shist[1]
    # bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hhist[0], shist[0], vhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient = 8, pix_per_cell = 9, cell_per_block = 2, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False, block_norm = 'L2-Hys')
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec, block_norm = 'L2-Hys')
        return features

img = mpimg.imread('data/vehicles/GTI_MiddleClose/image0091.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
c_feature = HSL_hist(img)
hog_feature = get_hog_features(gray)
# print(hog_feature.shape)
# %% train classifier




def extract_features(image, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, feature_vec = True):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    # Call get_hog_features() with vis=False, feature_vec=True

    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=feature_vec))
        hog_features = np.ravel(hog_features)
    elif hog_channel == 'gray':
        gray = cv2.cvtColor(feature_image, cv2.COLOR_RGB2GRAY)
        hog_features = get_hog_features(gray, orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=feature_vec)

    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=feature_vec)
    return hog_features

colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 6
pix_per_cell = 16
cell_per_block = 2
hog_channel = 0

def get_training_feature(img):
    # c_feature = HSL_hist(img)
    feature = extract_features(img, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
    # hog_features = get_hog_features(img[:,:,0], orient,
    #             pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    # feature = np.concatenate((c_feature, hog_feature))
    # feature = hog_features
    # print(feature.shape)
    return feature



def prepare_data():
    nocar_list = glob.glob('data/non-vehicles/**/*.png')
    nocar_list += glob.glob('test_patch/noncar/*.png')
    car_list = glob.glob('data/vehicles/**/*.png')
    X = []
    Y = [0]*(len(nocar_list)*2)
    for a_path in nocar_list:
        img = mpimg.imread(a_path)
        feature = get_training_feature(img)
        X.append(feature)
        img_flip = cv2.flip(img, 0)
        feature = get_training_feature(img_flip)
        X.append(feature)

    Y +=  [1]*len(car_list)
    for a_path in car_list:
        img = mpimg.imread(a_path)
        feature = get_training_feature(img)
        X.append(feature)
    return X, Y
# %%
nocar_list = glob.glob('data/non-vehicles/**/*.png')
img = mpimg.imread(nocar_list[0])
# %%
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

X, Y = prepare_data()
X, Y = shuffle(X, Y)
train_ratio = 0.95
train_slice = int(len(X)*train_ratio)
Xtrain = X[:train_slice]
Ytrain = Y[:train_slice]
Xtest = X[train_slice:]
Ytest = Y[train_slice:]
# Fit a per-column scaler
X_scaler = StandardScaler().fit(Xtrain)
# Apply the scaler to X
scaled_Xtrain = X_scaler.transform(Xtrain)
scaled_Xtest = X_scaler.transform(Xtest)
## %%
clf = LinearSVC(verbose = 1, max_iter = 3000)
clf.fit(scaled_Xtrain,Ytrain)
print(clf.score(scaled_Xtest, Ytest))
# %% extra test data
car_patches = glob.glob("test_patch/car/*.png")
nocar_patches = glob.glob("test_patch/nocar/*.png")
patch_features = []
for a_path in car_patches + nocar_patches:
    img = mpimg.imread(a_path)
    feature = get_training_feature(img)
    patch_features.append(feature)
scaled_Xpatch = X_scaler.transform(patch_features)
Ypatch = np.hstack((np.ones(len(car_patches)), np.zeros(len(nocar_patches))))
print(scaled_Xpatch.shape, Ypatch.shape)
print(clf.score(scaled_Xpatch, Ypatch))
# %% sliding window implementaion
test = mpimg.imread('test_images/test1.jpg')

def get_y_limit(w):
    src = np.float32( # in x, y
        [[710, 464],
         [1055, 689],
         [248, 689],
         [574, 464]])
    l1 = src[3]
    l2 = src[2]
    r1 = src[0]
    r2 = src[1]
    n1 = (w - r1[0] + l1[0])*(r2[1]-r1[1])*(l2[1]-l1[1])
    n2 = r1[1]*(r2[0]-r1[0])*(l2[1]-l1[1])
    n3 = l1[1]*(l1[0]-l2[0])*(r2[1]-r1[1])
    d1 = (r2[0]-r1[0])*(l2[1]-l1[1])
    d2 = (l1[0]-l2[0])*(r2[1]-r1[1])
    y = (n1 + n2 + n3)/(d1 + d2)
    return y

def window_proposal(scale = 1, overlap = 0.5):
    # base size is 64x64
    # Assume cars in the test_images.
    # Assume car cannot be wider than lane but must wider than half the width of the lane
    # if the size of a box is w, ymax
    # ymin and ymax is the bottom of the bounding box

    box_w = int(64*scale)

    ymin = get_y_limit(box_w*1.3)
    ymax = get_y_limit(box_w*2.5)

    boxes = []
    step = (1 - overlap)*box_w
    xmin = 400
    xmax = 1280.0

    for x in np.arange(xmin, xmax - box_w, step):
        for y in np.arange(ymin, ymax, step):
            # x, y is the bottom left coner. Use top-left and bottom-right to represent bbox
            boxes.append( ((int(x), int(y-box_w)), (int(x+box_w), int(y))) )
    return boxes


def visualize_bbox(img, bboxes):
    img_draw = img.copy()
    for bbox in bboxes:
        img_draw = cv2.rectangle(img_draw, bbox[0], bbox[1], (0, 255, 0), 3)
    return img_draw

scale = 3
bboxes = window_proposal(scale, 0.8)
test = cv2.resize(test, (1280, 720))
test_bbox = visualize_bbox(test, bboxes)
plt.figure(figsize = (16, 9))
plt.imshow(test_bbox)
plt.show()

# %%
test = mpimg.imread('test_images/test1.jpg')

for s in range(1,5):
    vis_proposal = window_proposal(s, 0.75)
    test_bbox = visualize_bbox(test, vis_proposal)
    plt.imsave('proposal_{}.png'.format(s), test_bbox)
# plt.figure(figsize = (16, 9))
# plt.imshow(test_bbox)
# plt.show()
# %%
def shrink_bbox(bbox, scale):
    return ((bbox[0][0]//scale, bbox[0][1]//scale), (bbox[1][0]//scale, bbox[1][1]//scale))

def get_hog_feature_frame(frame):
    hog_feature = extract_features(frame, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, feature_vec = False)
    return hog_feature


def get_bbox(frame): # output bounding boxes
    bboxes = []
    scales = [1,2,3]
    patch_count = 0
    for s in scales:
        # scale the entire frame for efficiency
        small_frame = cv2.resize(frame, (frame.shape[1]//s, frame.shape[0]//s))
        small_frame = small_frame.astype(np.float32)/255
        proposals = window_proposal(s, 0.8)
        # hog_feature_frame = get_hog_feature_frame(small_frame)
        # print(hog_feature_frame.shape)
        # compute features
        for bbox in proposals:
            small_bbox = shrink_bbox(bbox, s)
            patch = small_frame[small_bbox[0][1]:small_bbox[1][1], small_bbox[0][0]:small_bbox[1][0]]
            # hog_box = shrink_bbox(small_bbox, pix_per_cell)
            # print(hog_box)
            # hog_feature_patch = hog_feature_frame[hog_box[0][1]:hog_box[0][1]+7, hog_box[0][0]:hog_box[0][0]+7]
            # print('hog_feature_patch', hog_feature_patch.shape , hog_feature_patch.ravel().shape)
            # hog_feature_vec = hog_feature_patch.ravel()
            # X = [hog_feature_vec]
            # print("small_frame", patch.shape, patch.dtype)
            # mpimg.imsave("patch6_{}.png".format(patch_count), patch)
            # patch = mpimg.imread("patch{}.png".format(patch_count))
            # print(patch.shape, patch.dtype)
            patch_count += 1
            x = get_training_feature(patch)
            # print((x-hog_feature_vec).sum())
            X = [x]
            X_scale = X_scaler.transform(X)
            Y = clf.predict(X_scale)
            if Y[0] == 1:
                bboxes.append(bbox)
    return bboxes
test = mpimg.imread('test_images/test4.jpg')
bboxes = get_bbox(test)
test_bbox = visualize_bbox(test, bboxes)
plt.figure(figsize = (16, 9))
plt.imshow(test_bbox)
plt.imsave("bboxes.png", test_bbox)
plt.show()

## %% aggregation
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

from scipy.ndimage.measurements import label

heatmap = np.zeros_like(test)
heatmap = add_heat(heatmap, bboxes)
plt.figure(figsize = (16, 9))
plt.imshow(heatmap.astype(np.uint8)*20, cmap='gray')
plt.savefig("heatmap.png")
plt.show()
heatmap = apply_threshold(heatmap, 5)
labels = label(heatmap)
print(labels[1], 'cars found')
print(labels[0].shape, labels[0].dtype)
plt.figure(figsize = (16, 9))
plt.imshow(labels[0].astype(np.uint8)*50, cmap='gray')
plt.savefig("labels.png")
plt.show()
## %% output

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Draw bounding boxes on a copy of the image
draw_img = draw_labeled_bboxes(np.copy(test), labels)
# Display the image
plt.figure(figsize = (16, 9))
plt.imshow(draw_img)
plt.imsave("result.png", draw_img)
plt.show()

# %% process image
def process_image(img, threshold = 7, heatmap = np.zeros_like(img)):
    bboxes = get_bbox(img)
    heatmap = add_heat(heatmap, bboxes)
    heatmap = apply_threshold(heatmap, threshold)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

img = mpimg.imread('test_images/test6.jpg')
output = process_image(img)
plt.imshow(output)
plt.show()
# %% process video
import imageio
import progressbar
bar = progressbar.ProgressBar()
def process_video(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    threshold = 10
    decay_ratio = 0.75
    init_heat = 0
    vid_shape = vid.get_meta_data()['size']
    heatmap = np.zeros((vid_shape[1], vid_shape[0])) + init_heat
    writer = imageio.get_writer("out_thres{}_decay{}_initheat{}_{}".format(threshold, decay_ratio, init_heat, filename), fps=15)
    writer_heat = imageio.get_writer("heat{}_decay{}_initheat{}_{}".format(threshold, decay_ratio, init_heat, filename), fps=15)
    for image in bar(vid):
        result = process_image(image, threshold, heatmap)
        heatmap *= decay_ratio
        writer_heat.append_data((heatmap*10).astype(np.uint8))
        writer.append_data(result)
process_video("project_video.mp4")
