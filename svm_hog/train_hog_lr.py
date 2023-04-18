from skimage.feature import hog
#from skimage.io import imread
import joblib,glob,os,cv2

from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
import numpy as np 
from sklearn.preprocessing import LabelEncoder

train_data = []
train_labels = []

pos_im_path = 'INRIAPerson/cutting/train_cutting/pos'
neg_im_path = 'INRIAPerson/64X128_neg'

model_path = 'svm_hog/model/lr_model.dat'

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path,"*.png")):
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,128))
    fd = hog(img,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    train_data.append(fd)
    train_labels.append(1)
# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path,"*.png")):
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,128))
    fd = hog(img,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    train_data.append(fd)
    train_labels.append(0)

train_data = np.float32(train_data)
train_labels = np.array(train_labels)
print('Data Prepared........')
print('Train Data:',len(train_data))
print('Train Labels (1,0)',len(train_labels))
print("""Classification with Logistic Regression""")

model = LogisticRegression()
print('Training...... Logistic Regression')
print(train_data)
print(train_labels)
model.fit(train_data,train_labels)
joblib.dump(model, model_path)
print('Model saved : {}'.format(model_path))