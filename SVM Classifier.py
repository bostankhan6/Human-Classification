from skimage.feature import hog
from skimage import io
import os
from sklearn.svm import SVC
import pickle

pos_path = './INRIA_Dataset_Samples/Train/pos_resized/'
neg_path = './INRIA_Dataset_Samples/Train/neg_resized/'

trainlist_pos = os.listdir(pos_path)
trainlist_neg = os.listdir(neg_path)

trainimages = []
labels = []

for file in trainlist_pos:
    image = io.imread((pos_path + file))
    fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    trainimages.append(fd)
    labels.append(1)

for file in trainlist_neg:
    image = io.imread((neg_path + file))
    fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    trainimages.append(fd)
    labels.append(0)

clf = SVC(gamma='auto')

print(len(trainimages))
print(len(labels))

clf.fit(trainimages, labels)

model_name = 'SVM Model.sav'
pickle.dump(clf, open(model_name, 'wb'))