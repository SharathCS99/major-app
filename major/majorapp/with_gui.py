import tkinter as tk
root=tk.Tk()
root.geometry("600x400")
name_var=tk.StringVar()
s=''
def submit():
    global s1
    s1=name_var.get()
    print("this ")
    print(s1)
    k=[]
    print(k)
    for i in range(0,len(s1)):
        if(ord(s1[i])==92):
            k.append('/')
        k.append(s1[i])
    print(k)
    lim=0
    print(k[len(k)-1],k[len(k)-2],k[len(k)-3])
    if(k[len(k)-1]!='4'):
        print("hertm")
        lim=1
    kc=[]
    k_copy=0
    while(k_copy<len(k)):
        if(ord(k[k_copy])==92):
            k_copy=k_copy+1
        else:
            kc.append(k[k_copy])
            k_copy+=1
    print(kc)
    print("here0")
    path_op=''.join(kc)
    print(path_op)
    global s
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from local_utils import detect_lp
    from os.path import splitext,basename
    from keras.models import model_from_json
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.mobilenet_v2 import preprocess_input
    from sklearn.preprocessing import LabelEncoder
    import glob
    s=''
    cap = cv2.VideoCapture(path_op)
    success,image = cap.read()
    count = 0
    while success:
        print("here")
        cv2.imwrite("C:/Users/admin/OneDrive/Desktop/Project/Plate_detect_and_recognize-master/Plate_examples/frames/img"+str(count)+".jpg", image)     # save frame as JPEG file      
        success,image = cap.read()
        print('Read a new frame: ', success)
        count += 1
        if(count==200):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("broke")
    def load_model(path):
        try:
            path = splitext(path)[0]
            with open('%s.json' % path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json, custom_objects={})
            model.load_weights('%s.h5' % path)
            print("Loading model successfully...")
            return model
        except Exception as e:
            print(e)

    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)
    def preprocess_image(image_path,resize=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if resize:
            img = cv2.resize(img, (224,224))
        return img

    def get_plate(image_path, Dmax=608, Dmin = 608):
        vehicle = preprocess_image(image_path)
        ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
        return vehicle, LpImg, cor
    local_max=0
    s=''
    i1=count
    print(count)
    tm=-1
    if(lim==1):
        tm=i1-1
    for maininde in range(i1,tm,-1):
        print(maininde)
        test_image_path = "Plate_examples/frames/img"+str(maininde)+".jpg"
        if(i1-1==tm):
            print("hereiop")
            test_image_path = path_op
        try:
            vehicle, LpImg,cor = get_plate(test_image_path)

            fig = plt.figure(figsize=(12,6))
            grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
            fig.add_subplot(grid[0])
            plt.axis(False)
            plt.imshow(vehicle)
            grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
            fig.add_subplot(grid[1])
            plt.axis(False)
            plt.imshow(LpImg[0])
            if (len(LpImg)): 
                plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(7,7),0)
                binary = cv2.threshold(blur, 180, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            fig = plt.figure(figsize=(12,7))
            plt.rcParams.update({"font.size":18})
            grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
            plot_image = [plate_image, gray, blur, binary,thre_mor]
            plot_name = ["plate_image","gray","blur","binary","dilation"]
            for i in range(len(plot_image)):
                fig.add_subplot(grid[i])
                plt.axis(False)
                plt.title(plot_name[i])
                if i ==0:
                    plt.imshow(plot_image[i])
                else:
                    plt.imshow(plot_image[i],cmap="gray")
            def sort_contours(cnts,reverse = False):
                i = 0
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
                return cnts
            cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            test_roi = plate_image.copy()
            crop_characters = []


            digit_w, digit_h = 30, 60
    
            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1<=ratio<=3.5:
                    if h/plate_image.shape[0]>=0.5:
                        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)

            print("Detect {} letters...".format(len(crop_characters)))
            fig = plt.figure(figsize=(10,6))
            plt.axis(False)
            plt.imshow(test_roi)

            fig = plt.figure(figsize=(14,4))
            json_file = open('MobileNets_character_recognition.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights("License_character_recognition_weight.h5")
            print("Model loaded successfully...")

            labels = LabelEncoder()
            labels.classes_ = np.load('license_character_classes.npy')
            print("Labels loaded successfully...")
            grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

            for i in range(len(crop_characters)):
                fig.add_subplot(grid[i])
                plt.axis(False)
                plt.imshow(crop_characters[i],cmap="gray")
            def predict_from_model(image,model,labels):
                image = cv2.resize(image,(80,80))
                image = np.stack((image,)*3, axis=-1)
                prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
                return prediction
            fig = plt.figure(figsize=(15,3))
            cols = len(crop_characters)
            grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
            final_string = ''
            for i,character in enumerate(crop_characters):
                fig.add_subplot(grid[i])
                title = np.array2string(predict_from_model(character,model,labels))
                plt.title('{}'.format(title.strip("'[]"),fontsize=20))
                final_string+=title.strip("'[]")
                plt.axis(False)
                plt.imshow(character,cmap='gray')
            if(local_max<len(crop_characters)):
                local_max=len(crop_characters)
                s=final_string
            if(local_max>=9):
                break
        except:
            continue
    print(s)
    name_labe = tk.Label(root, text="THIS IS THE NUMBER "+s, font=('calibre',10, 'bold'))
    name_labe.place(x=100,y=200,relx=0.25,rely=0.25)
name_label = tk.Label(root, text = 'Give the Video Path', font=('calibre',10, 'bold'))
name_entry = tk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'))
sub_btn=tk.Button(root,text = 'Submit', command = submit)
print(s)
name_labe = tk.Label(root, text=s, font=('calibre',10, 'bold'))
name_label.place(x=50,y=50,relx=0.25,rely=0.25)
name_entry.place(x=150,y=50,relx=0.25,rely=0.25)
sub_btn.place(x=100,y=75,relx=0.25,rely=0.25)
root.mainloop()


