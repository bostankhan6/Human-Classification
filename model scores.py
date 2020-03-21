from skimage.feature import hog
from skimage import io
import os
from pandas_confusion import ConfusionMatrix
from sklearn import metrics
import pickle

model_name = 'RandomForest.sav'
clf = pickle.load(open(model_name, 'rb'))

pos_path = './INRIA_Dataset_Samples/Test/pos_resized/'
neg_path = './INRIA_Dataset_Samples/Test/neg_resized/'

testlist_pos = os.listdir(pos_path)
testlist_neg = os.listdir(neg_path)

testimages = []
labels = []

for file in testlist_pos:
    image = io.imread((pos_path + file))
    fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    testimages.append(fd)
    labels.append(1)

for file in testlist_neg:
    image = io.imread((neg_path + file))
    fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    testimages.append(fd)
    labels.append(0)

print(len(testimages))
print(len(labels))

predicted_labels = clf.predict(testimages)

#print(clf.score(testimages,labels))

print("Precision Score: %4.2f" %(metrics.precision_score(labels,predicted_labels)))
print("Recall Score: %4.2f" %(metrics.recall_score(labels,predicted_labels)))
print("F1 Score: %4.2f" %(metrics.f1_score(labels,predicted_labels)))
print("Accuracy Score: %4.2f" %(metrics.accuracy_score(labels,predicted_labels)))

confusion_matrix = ConfusionMatrix(labels,predicted_labels)
print("Confusion matrix:\n%s" % confusion_matrix)


