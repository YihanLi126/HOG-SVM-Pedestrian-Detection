import glob,os,cv2

neg_counter=0
interval=64
# for filename in glob.glob(os.path.join('INRIAPerson/Train/neg/',"*.png")):
#     img = cv2.imread(filename)
#     x_range=int((img.shape[1]-64)/interval)
#     y_range=int((img.shape[0]-128)/interval)
#     for i in range (0,y_range):
#         for j in range(0,x_range):
#             cropped = img[interval*i:interval*i+128, interval*j:interval*j+64] 
#             cv2.imwrite("INRIAPerson/64X128_neg/neg_"+str(neg_counter)+".png", cropped)
#             neg_counter+=1

for filename in glob.glob(os.path.join('INRIAPerson/Train/neg/',"*.png")):
    img = cv2.imread(filename)
    for i in range (0,20):
        cropped = img[10*i:10*i+128, 10*i:10*i+64] 
        cv2.imwrite("INRIAPerson/64X128_neg2/neg_"+str(neg_counter)+".png", cropped)
        neg_counter+=1

