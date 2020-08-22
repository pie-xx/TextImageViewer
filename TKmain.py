#import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import BookSharpener
from PIL import Image, ImageTk

def getFitsize( w, h, sw, sh ):
    if w < h:
        vh = sh
        vw = sw * (w/h)
    else:
        vw = sw
        vh = sh * (h/w)
    return int(vw), int(vh)

convfile=""

def button1_clicked():
    global convfile
    file = filedialog.askopenfilename(initialdir='.')
    if file:
        img = Image.open(file)
        setImage( img )
        v1.set(file)
        convfile = file
        print("button1_clicked", convfile)
        root.after( 500, func=convImage )

def convImage():
    global convfile
    print("convImage", convfile)
    imgname = BookSharpener.sharpenImg(convfile)
    v1.set(imgname)
    img = Image.open(imgname)
    setImage( img )

def setImage( img ):
    vw, vh = getFitsize(img.width, img.height, 800, 800)
    img = img.resize((vw, vh))
    tkimg = ImageTk.PhotoImage(img)
    c1.photo=tkimg
    c1.itemconfig(c1id, image=tkimg)
    

root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

frame = ttk.Frame(root, padding=10)
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)
frame.grid(sticky=(N, W, S, E))

# Open File
b1 = ttk.Button(
     frame, text='Open File', width=15,
     command=button1_clicked)
b1.grid(row=0, column=0, sticky=(W))

v1 = StringVar()
l1 = ttk.Label(frame, textvariable=v1)
l1.grid(row=0, column=1)

c1 = Canvas(root, width=800, height=800, bg="white")
c1.grid(row=1, column=0)
img = Image.open("SIP/20200326012127794.jpg")
vw, vh = getFitsize(img.width, img.height, 800, 800)
img = img.resize((vw, vh))
tkimg = ImageTk.PhotoImage(img)
c1id = c1.create_image(400, 400, image=tkimg)

root.mainloop()
