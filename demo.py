import torch
from torchvision import transforms
from test_mmd import perform_test
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from functools import partial

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.225, 0.225, 0.225])])



#Create & Configure root 
root = Tk()
sub = Tk()
sub.title("Output")
root.title("Image Tampering Detection Demo")
a = perform_test(transform, sub)
Grid.rowconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 0, weight=1)

#Create & Configure frame 
frame=Frame(root)
frame.grid(row=0, column=0, sticky=N+S+E+W)

#Create a 5x2 (rows x columns) grid of buttons inside the frame
for col_index in range(2):
    Grid.columnconfigure(frame, col_index, weight=1)
    for row_index in range(5):
        Grid.rowconfigure(frame, row_index, weight=1)
        btn = Button(frame) #create a button inside frame 
        btn.grid(row=row_index, column=col_index, sticky=N+S+E+W) 
        if col_index == 0:
            img = ImageTk.PhotoImage(Image.open('data/au/img_'+ str(row_index + 1)+'.png').resize((300,150),Image.ANTIALIAS))
            btn.config(image = img, command = partial(a, 'data/au/img_'+ str(row_index + 1)+'.png'))
            btn.image = img
        else:
            img = ImageTk.PhotoImage(Image.open('data/tp/img_'+ str(5+ row_index + 1)+'.png').resize((300,150),Image.ANTIALIAS))
            btn.config(image = img, command = partial( a, 'data/tp/img_'+ str(5+row_index + 1)+'.png'))
            btn.image = img
        
        
root.mainloop()

