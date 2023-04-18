import glob,os,cv2

pos_counter=0
for filename in glob.glob(os.path.join('INRIAPerson/Train/pos/',"*.png")):
    img = cv2.imread(filename)
    # if img is None:
    #     print("Train/pos is None!")
    #     continue
    # else:
    #     print("pos",pos_counter," read sucessfully !")
    width=img.shape[0]
    height=img.shape[1]
    print(img.shape)
    ys=int(height/2)
    xs=int(width/2)
    cropped = img[ys-80:ys+80, xs-48:xs+48] 
    if cropped is not None:
        cv2.imwrite("INRIAPerson/96X160_pos/pos_"+str(pos_counter)+".png", cropped)
        print("pos img saved")
        pos_counter+=1
    