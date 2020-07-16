#import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import BookSharpener

def button1_clicked():
    file = filedialog.askopenfilename(initialdir='.')
    if file:
        v1.set(file)
        BookSharpener.sharpenImg(file)

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



root.mainloop()
