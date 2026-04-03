import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('D:\\project\\face-recognition-emoji\\model.weights.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


emoji_dist = {
    0: r"D:\project\face-recognition-emoji\emojis\angry.png",
    1: r"D:\project\face-recognition-emoji\emojis\disgusted.png",
    2: r"D:\project\face-recognition-emoji\emojis\fearful.png",
    3: r"D:\project\face-recognition-emoji\emojis\happy.png",
    4: r"D:\project\face-recognition-emoji\emojis\neutral.png",
    5: r"D:\project\face-recognition-emoji\emojis\sad.png",
    6: r"D:\project\face-recognition-emoji\emojis\surprised.png"
}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
cap1 = cv2.VideoCapture(0)                                 
show_text=[0]
def show_vid():      
    if not cap1.isOpened():                             
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))

    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def show_vid2():
    # 1. Get the path from your dictionary
    img_path = emoji_dist[show_text[0]]
    frame2 = cv2.imread(img_path)

    # 2. THE CRITICAL SAFETY CHECK
    if frame2 is None:
        print(f"Error: Could not find image at {img_path}. Check your file paths!")
        # Instead of crashing, we wait and try again
        lmain2.after(100, show_vid2) 
        return

    # 3. Convert BGR (OpenCV) to RGB (PIL/Tkinter)
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # 4. Create the Tkinter-compatible image
    img2 = Image.fromarray(pic2) # Use pic2 here, not frame2!
    imgtk2 = ImageTk.PhotoImage(image=img2)
    
    # 5. Update the GUI
    lmain2.imgtk2 = imgtk2
    lmain2.configure(image=imgtk2)
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    
    # 6. Repeat
    lmain2.after(10, show_vid2)

if __name__ == '__main__':
    root=tk.Tk()   
    img = ImageTk.PhotoImage(Image.open("logo.webp"))
    heading = Label(root,image=img,bg='black')
    
    heading.pack() 
    heading2=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
    
    heading2.pack()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)

    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=20, y=50)
    lmain3.pack()
    lmain3.place(x=550, y=300)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=550, y=50)


    root.title("Photo To Emoji")            
    root.geometry("800x500+100+10")
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    show_vid()
    show_vid2()
    root.mainloop()