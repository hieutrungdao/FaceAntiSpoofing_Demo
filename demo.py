import cv2
import numpy as np
import lightgbm as lgb
from skimage import feature as skif


dim = (32, 32)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load lightgbm weight
clf = lgb.Booster(model_file='lgb.txt')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')


def lbp_histogram(image, P=8, R=1, method='nri_uniform'):
    '''
    image: shape is N*M 
    '''
    lbp = skif.local_binary_pattern(
        image, P, R, method)  # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    max_bins = int(lbp.max() + 1)  # max_bins is related P
    hist, _ = np.histogram(
        lbp,  normed=True, bins=max_bins, range=(0, max_bins))
    return hist


def save_features(images):
    list_feature = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = skif.local_binary_pattern(gray, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, 24 + 3),
        range=(0, 24 + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_h = lbp_histogram(image[:,:,0]) # y channel
        cb_h = lbp_histogram(image[:,:,1]) # cb channel
        cr_h = lbp_histogram(image[:,:,2]) # cr channel
        feature = np.concatenate((y_h, cb_h, cr_h, hist))
        list_feature.append(feature)
    return list_feature


while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        lbp = skif.local_binary_pattern(gray, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, 24 + 3),
        range=(0, 24 + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        y_h = lbp_histogram(face[:, :, 0])  # y channel
        cb_h = lbp_histogram(face[:, :, 1])  # cb channel
        cr_h = lbp_histogram(face[:, :, 2])  # cr channel

        feature = np.concatenate((y_h, cb_h, cr_h, hist)).reshape(1, -1)

        prediction = clf.predict(feature, num_iteration=clf.best_iteration)

        print(prediction)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        text = str()

        if (prediction > 0.01):
            text = "fake"
        else:
            text = "real"

        cv2.putText(img, text+str(prediction), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
