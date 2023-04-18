from skimage.feature import hog
import joblib
import numpy as np
import cv2
from sklearn import *
import os
import glob
from sklearn.metrics import * #roc_curve, auc
import matplotlib.pyplot as plt

model_svm = joblib.load('svm_hog/model/svm_model.dat')
model_lr = joblib.load('svm_hog/model/lr_model.dat')

test_data=[]
test_labels=[]
pos_im_path = 'INRIAPerson/Test/pos'
neg_im_path = 'INRIAPerson/Test/neg'

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path,"*.*")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    test_data.append(fd)
    test_labels.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path,"*.*")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    test_data.append(fd)
    test_labels.append(0)
test_data = np.float32(test_data)
test_labels = np.array(test_labels)

# For ROC and AUC
Y_pred_svm = model_svm.predict(test_data)
Y_score_svm = model_svm.decision_function(test_data)
Y_pred_lr = model_lr.predict(test_data)
Y_score_lr = model_lr.decision_function(test_data)

fpr_svm, tpr_svm, thresholds_svm = roc_curve(test_labels, Y_score_svm, pos_label=1)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(test_labels, Y_score_lr, pos_label=1)


auc_svm = roc_auc_score(test_labels, Y_score_svm)
auc_lr = roc_auc_score(test_labels, Y_score_lr)
print("AUC of SVM is: ",auc_svm)
print("AUC of LR is: ",auc_lr)

# For mr Vs fpr
svm_sort_index=np.argsort(Y_score_svm)
lr_sort_index=np.argsort(Y_score_lr)
# print("svm_sort_index: ",svm_sort_index)
# print("lr_sort_index: ",lr_sort_index)
sum = len(test_labels)
mr_svm=np.zeros(sum)
fpr2_svm=np.zeros(sum)
mr_lr=np.zeros(sum)
fpr2_lr=np.zeros(sum)

for i in range(30,sum):
    svm_label=np.zeros(i)
    lr_label=np.zeros(i)
    svm_pred=np.zeros(i)
    lr_pred=np.zeros(i)
    for index in range(0,i):
        svm_label[index]=test_labels[svm_sort_index[index]]
        lr_label[index]=test_labels[lr_sort_index[index]]
        svm_pred[index]=Y_pred_svm[svm_sort_index[index]]
        lr_pred[index]=Y_pred_lr[lr_sort_index[index]]
    # print("svm_label: ",svm_label)
    # print("svm_pred: ",svm_pred)
    # if len(svm_label)==0:
    #     mr_svm[i]=0
    #     mr_lr[i]=0
    #     fpr2_svm[i]=0
    #     fpr2_lr[i]
    #     continue
    # elif np.sum(svm_label)==0 and np.sum(svm_pred)==0:
    #     mr_svm[i]=0
    #     mr_lr[i]=0
    #     fpr2_svm[i]=0
    #     fpr2_lr[i]
    #     continue
    tn_svm, fp_svm, fn_svm, tp_svm  = confusion_matrix(svm_label, svm_pred).ravel()
    tn_lr, fp_lr, fn_lr, tp_lr  = confusion_matrix(lr_label,lr_pred).ravel()
    mr_svm[i]=tp_svm/(tp_svm+fn_svm)
    mr_lr[i]=tp_lr/(tp_lr+fn_lr)
    fpr2_svm[i]=fp_svm/(tn_svm+fp_svm)
    fpr2_lr[i]=fp_lr/(tn_lr+fp_lr)

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(18,6)
ax_mr_fpr=ax[0]
ax_roc=ax[1]


ax_mr_fpr.plot(mr_svm,fpr2_svm,'b',label="SVM MR-FPR")
ax_mr_fpr.plot(mr_lr,fpr2_lr,'r',label="LR MR-FPR")
ax_mr_fpr.set_xlabel("False Posotive Rate", fontsize=12, fontname="Times New Roman")
ax_mr_fpr.set_ylabel("Missing Rate", fontsize=12, fontname="Times New Roman")

ax_roc.plot(fpr_svm, tpr_svm, 'b',label='SVM ROC')
ax_roc.plot(fpr_lr,tpr_lr,'r',label='LR ROC')
ax_roc.set_xlabel("False Posotive Rate", fontsize=12, fontname="Times New Roman")
ax_roc.set_ylabel("True Posotive Rate", fontsize=12, fontname="Times New Roman")

for tick in ax_roc.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax_roc.get_yticklabels():
    tick.set_fontname("Times New Roman")

for tick in ax_mr_fpr.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax_mr_fpr.get_yticklabels():
    tick.set_fontname("Times New Roman")

ax_mr_fpr.axis('equal')
ax_roc.axis('equal')
ax_roc.legend(loc="lower right")
ax_mr_fpr.legend(loc="lower right")

plt.savefig(
    "svm_hog/plot/test_curvs.png",
    format="png",
    dpi=1000,
    pad_inches=0,
    bbox_inches="tight",
)

plt.show()

