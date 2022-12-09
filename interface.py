import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from modelProjet import prediction

interface = tk.Tk()
interface.geometry("800x400")  # Size of the window 
interface.title('Mini Projet : Kalel-Landolsi-Guesmi')
font_header=('times', 18, 'bold')
espace = tk.Label(interface,text=' ',width=30,font=font_header)  
espace.grid(row=1,column=1,columnspan=10)

heade = tk.Label(interface,text='Upload Files & display',width=30,font=font_header)  
heade.grid(row=2,column=1,columnspan=10)
btn_upload_files = tk.Button(interface, text='Upload Files', 
   width=20,command = lambda:upload_file(), bg="#154360", fg="white")
btn_upload_files.grid(row=3,column=1,columnspan=10)

espace = tk.Label(interface,text=' ',width=30,font=font_header)  
espace.grid(row=4,column=1,columnspan=10)


def upload_file():
    file_Types = [('Jpg Files', '*.png'),
    ('PNG Files','*.png')]   # type of files to select 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=file_Types)
    column=1 # start from columnumn 1
    row=5 # start from row 3 
    for f in filename:
        img=Image.open(f) # read the image file
        img=img.resize((306,306)) # new width & height
        img=ImageTk.PhotoImage(img)
        imageBlocks = tk.Label(interface)
        imageBlocks.grid(row=row,column=column)
        result_prediction = tk.Label(interface,text=prediction(f),font=('times', 15, 'bold'))
        result_prediction.grid(row=row+1,column=column,columnspan=1)
        imageBlocks.image = img
        imageBlocks['image']=img # garbage columnlection 
        if(column==6): # start new line after third columnumn
            row=row+2# start wtih next row
            column=1    # start with first columnumn
        else:       # within the same row 
            column=column+1 # increase to next columnumn                
interface.mainloop()  # Keep the window open