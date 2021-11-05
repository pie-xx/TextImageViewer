#横または縦のライン方向についてスキャンした濃淡グラフ
# 画像を選択表示する。表示画像をクリックすると近傍64dot四方のヒストグラムを表示
#
import cv2
import tkinter
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

import pyocr
import pyocr.builders

from sklearn.cluster import KMeans

Vwidth = 900
Vheight = 900

reso=16
ax=0
ay=0
xbias=0
ybias=0
smap = []

thr=10
splen=32
avw=4

class CAPapp():
    def __init__(self, **kwargs):
        self.root = tkinter.Tk()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0,weight=1)

        self.frame=ttk.Frame(self.root,padding=10)
        self.frame.columnconfigure(0,weight=1)
        self.frame.rowconfigure(0,weight=1)
        self.frame.grid(sticky=(tkinter.N,tkinter.W,tkinter.S,tkinter.E))

        self.fbar=ttk.Frame(self.frame,padding=4)
        self.fbar.columnconfigure(0,weight=1)
        self.fbar.rowconfigure(0,weight=1)
        self.fbar.grid(sticky=(tkinter.N,tkinter.W,tkinter.S,tkinter.E))

        self.button = ttk.Button(
                self.fbar, text="Open", width=6,
                command=self.fileopen_clicked
                )
        self.button.grid(row=0,column=3)

        self.text = tkinter.StringVar()
        self.text.set("X")
        self.label = ttk.Label(self.fbar, textvariable=self.text)
        self.label.grid(row=0,column=0)

        self.canvas=tkinter.Canvas(self.root, width=Vwidth, height=Vheight, bg='white')
        self.canvas.grid(row=2,column=0)

        self.canvas.bind('<ButtonPress-1>', self.clickImg)

        self.img = []
        self.cnvimg = []

        # 2.OCRエンジンの取得
        tools = pyocr.get_available_tools()
        self.tool = tools[0]

    def sliceScanX(self, px, py, csy, cey):

        #scan=[]
        #for x in range(self.img.shape[1]-avw):
        #    limg = self.img[csy:cey,x:x+avw]
        #    scan.append(np.std(limg))
        #plt.plot( range(self.img.shape[1]-avw), scan)

        csy, cey = self.scanareaY(py)
        #scan=[]
        minx = 0
        cnt=0
        for x in range(px,0,-1):
            limg = self.img[csy:cey,x:x+avw]
            std = np.std(limg)
            if( std < thr ):
                minx = x
                cnt=cnt+1
                if cnt > splen:
                    break
            else:
                cnt=0
            #scan.insert(0,std)

        cnt=0
        maxx=self.img.shape[1]
        for x in range(px,self.img.shape[1]-avw):
            limg = self.img[csy:cey,x:x+avw]
            std = np.std(limg)
            if( std < thr ):
                maxx = x
                cnt=cnt+1
                if cnt > splen:
                    break
            else:
                cnt=0
            #scan.append(std)
        #plt.show()
        return minx, maxx
    
    def sliceScanY(self, px, py, csx, cex):

        #scan=[]
        #for y in range(self.img.shape[0]-avw):
        #    limg = self.img[y:y+avw,csx:cex]
        #    scan.append(np.std(limg))
        #plt.plot( scan, range(self.img.shape[0]-avw) )

        #scan=[]
        miny = 0
        cnt=0
        for y in range(py,0,-1):
            limg = self.img[y:y+avw,csx:cex]
            std = np.std(limg)
            if( std < thr ):
                miny = y
                cnt=cnt+1
                if cnt > splen:
                    break
            else:
                cnt=0
            #scan.insert(0,std)

        cnt=0
        maxy=self.img.shape[0]
        for y in range(py,self.img.shape[0]-avw):
            limg = self.img[y:y+avw,csx:cex]
            std = np.std(limg)
            if( std < thr ):
                maxy = y
                cnt=cnt+1
                if cnt > splen:
                    break
            else:
                cnt=0
            #scan.append(std)
        #plt.show()
        return miny, maxy

    def clickImg(self, event):
        _px, _py = self.scr2pic(event.x,event.y)
        print(_px,_py)
        px = int(_px)
        py = int(_py)

        ori = self.localscan(px,py)
        tx,ty = event.x,event.y

        if ori=="H":
            csy, cey = self.scanareaY(py)
            minx, maxx = self.sliceScanX(px,py,csy,cey)            
            print("minx,maxx:", minx, maxx)
            
            sx, sy=self.pic2scr(minx,csy)
            ex, ey=self.pic2scr(maxx,cey)
            self.canvas.create_rectangle(sx, sy, ex, ey, outline='red' )
            self.canvas.create_rectangle(tx-4,ty-4,tx+4,ty+4,outline="red")
        
            spfrm = int(splen *1.5)
            miny = py
            headx = int(((maxx-minx)/4)+minx)
            for y in range(py,0,-spfrm):
                limg = self.img[y:y+spfrm,minx:headx]
                limstd = np.std(limg)
                miny = y
                if limstd < thr:
                    break

            for y in range(py,self.img.shape[0],spfrm):
                limg = self.img[y:y+spfrm,minx:headx]
                limstd = np.std(limg)
                maxy = y
                if limstd < thr:
                    break

            sx, sy=self.pic2scr(minx,miny)
            ex, ey=self.pic2scr(maxx,maxy+splen)
            self.canvas.create_rectangle(sx, sy, ex, ey, outline='blue' )

        if ori=="V": #縦書き
            csx, cex = self.scanareaX(px)
            miny, maxy = self.sliceScanY(px,py,csx,cex)
            print("miny,maxy:", miny, maxy)
            
            sx, sy=self.pic2scr(csx,miny)
            ex, ey=self.pic2scr(cex,maxy)
            self.canvas.create_rectangle(sx, sy, ex, ey, outline='red' )
            self.canvas.create_rectangle(tx-4,ty-4,tx+4,ty+4,outline="red")
        
            spfrm = int(splen *1.5)
            minx = px
            heady = int(((maxy-miny)/4)+miny)
            for x in range(px,0,-spfrm):
                limg = self.img[miny:heady,x:x+spfrm]
                limstd = np.std(limg)
                minx = x
                if limstd < thr:
                    break

            for x in range(px,self.img.shape[1],spfrm):
                limg = self.img[miny:heady,x:x+spfrm]
                limstd = np.std(limg)
                maxx = x
                if limstd < thr:
                    break

            sx, sy=self.pic2scr(minx,miny)
            ex, ey=self.pic2scr(maxx,maxy+splen)
            self.canvas.create_rectangle(sx, sy, ex, ey, outline='blue' )
                       

    def getFitsize(self, w, h, sw, sh ):
        if w < h:
            vh = sh
            vw = sw * (w/h)
        else:
            vw = sw
            vh = sh * (h/w)
        return int(vw), int(vh)
            
    def setImg( self ):
        global ax,ay,xbias,ybias

        self.canvas.delete("all")
    
        vw, vh = self.getFitsize(self.img.shape[1], self.img.shape[0], Vwidth, Vheight)
        self.vh = vh
        self.vw = vw

        img = cv2.resize(self.img , (vw, vh))
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgbimg)
        self.tkimg = ImageTk.PhotoImage(pil_image)
        self.c1id=self.canvas.create_image(Vwidth/2, Vheight/2, image=self.tkimg)

        if self.img.shape[1] > self.img.shape[0]:
            ax = Vwidth / self.img.shape[1]
            ay = self.vh / self.img.shape[0]
            ybias = (Vheight - self.vh )/2
            xbias=0
        else:
            ax = self.vw / self.img.shape[1]
            ay = Vheight / self.img.shape[0]
            xbias = (Vwidth - self.vw )/2
            ybias = 0

    def scanareaX(self, cx):
        scanwidth = 128
        csx = cx - scanwidth
        if csx < 0:
            csx = 0
        cex = cx + scanwidth
        if cex > self.img.shape[1]:
            cex = self.img.shape[1]
        return csx, cex

    def scanareaY(self, cy):
        scanheight = 128
        csy = cy - scanheight
        if csy < 0:
            csy = 0
        cey = cy + scanheight
        if cey > self.img.shape[0]:
            cey = self.img.shape[0]
        return csy, cey

    def orientationCheck(self, csx, csy, cex, cey):
        cimg = self.img[csy:cey, csx:cex]
        strx = [0]*(cex-csx)
        for n in range((cex-csx)):
            strx[n]=np.std( cimg[n:n+1,0:(cex-csx)])
        stry = [0]*((cey-csy))
        for n in range((cey-csy)):
            stry[n]=np.std( cimg[0:(cey-csy),n:n+1])
        
        stdx = np.std(strx)
        stdy = np.std(stry)
        if stdx > stdy:
            return "H", strx
        return  "V", stry

    def localscan(self, _cx, _cy):
        global thr, splen
        cx = int(_cx)
        cy = int(_cy)

        csx, cex = self.scanareaX(cx)
        csy, cey = self.scanareaY(cy)

        ori, scanar = self.orientationCheck(csx, csy, cex, cey)

        mean = np.mean(scanar)
        lowlist = [i for i in scanar if i < mean]
        if( len(lowlist)==0):
            lmean=0
        else:
            lmean = np.mean(lowlist)
        thr = (mean+lmean)/2
        spws=[]
        chws=[]
        cnt=0
        bcnt=0
        for n in scanar:
            if( n < thr ):
                # 白地
                cnt = cnt + 1
                if bcnt !=0 :
                    chws.append(bcnt)
                    bcnt=0
            else:
                # 文字
                bcnt = bcnt + 1
                if cnt != 0:
                    spws.append(cnt)
                    cnt = 0

        if len(chws)!=0:
            splen = np.max(chws)
        else:
            splen = 32

        print("scanar mean lmean", mean,lmean)
        print("spws ", spws)            
        print("chws ", chws)
        print("splen ", splen )      
        print("orientation ", ori)

        #self.showFFT(scanar)

        return ori

    def showFFT(self, stry):
        F = np.fft.fft(stry)
        N = len(stry)

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,6))

        ax[0].plot(range(len(stry)), stry )
        ax[0].plot([0,len(stry)],[thr,thr],color="red")
        freq = np.fft.fftfreq(N,1.0/256)
        Amp = np.abs(F/(N/2)) # 振幅

        #print( "F", F[:10])
        #print( "Amp", Amp[:10])
        #print( "freq", freq[:10])

        ax[1].plot(freq[1:int(N/2)], Amp[1:int(N/2)])
        ax[1].set_xlabel("Freqency [Hz]")
        ax[1].set_ylabel("Amplitude")
        ax[1].grid()

        #ax.plot(freq)

        pass


    def pic2scr(self, px, py):
        return  px*ax+xbias, py*ay+ybias

    def scr2pic(self, sx, sy):
        return  (sx-xbias)/ax, (sy-ybias)/ay

    def fileopen_clicked(self):
        filename = filedialog.askopenfilename(initialdir='.')
        if filename:
            with open( filename, 'rb') as f:
                fdata =f.read()
                inp = np.frombuffer(fdata, dtype = 'int8')
                self.img = cv2.imdecode(inp, cv2.IMREAD_UNCHANGED)
                self.oimg = cv2.imdecode(inp,cv2.IMREAD_UNCHANGED)
                print( filename, self.img.shape[1], self.img.shape[0] )
                self.setImg()
                self.filename =  filename

    def stdscan(self):
        global smap
        smap = []
        for y in range(reso,self.img.shape[0],reso):
            lm = []
            for x in range(reso,self.img.shape[1],reso):
                cimg = self.img[ y-reso:y, x-reso:x ]
                s = np.std(cimg)
                #pimg.append( s )
                lm.append(s)
                if( s > 15):
                    self.canvas.create_rectangle((x-reso)*ax+xbias, (y-reso)*ay+ybias, x*ax+xbias, y*ay+ybias, fill='red', )
                    cv2.rectangle(self.oimg, (x-reso, y-reso), (x,y), (0,0,255), 1 )
            smap.append(lm)       

        cv2.imwrite("output.jpg",self.oimg)
        pass


    def run(self):
        self.root.mainloop()

    def rectangle( self, sx, sy, ex, ey, col ):
        self.canvas.create_rectangle(int((sx)*ax+xbias), int((sy)*ay+ybias), int(ex*ax+xbias), int(ey*ay+ybias), fill=col, )


app = CAPapp()
app.run()

