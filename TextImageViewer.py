#横または縦のライン方向についてスキャンした濃淡グラフ
# 画像を選択表示する。表示画像をクリックすると近傍64dot四方のヒストグラムを表示
#
import os
import json
from posixpath import basename

import cv2
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
import numpy as np
from PIL import Image, ImageTk

import BookSharpener

Vwidth = 1100
Vheight = 1600
AnnoSaveFile = ".annofilerx.json"
CpageSaveFile = ".annofilers.json"

reso=16
ax=0
ay=0
xbias=0
ybias=0
smap = []

thr=10
splen=32
avw=4

class AnnoFiler():
    def __init__(self, filename):
        self.initialdir = os.path.dirname(filename)
        self.aprof = json.loads('{"annos":{"'+AnnoSaveFile+'":{"text":"annotation"}}}')
        try:
            with open(self.initialdir+'/'+AnnoSaveFile,'r',encoding='utf-8') as f:
                self.aprof = json.load(f)
                annos = self.aprof['annos']
                for a in annos:
                    print(a,annos[a])
        except:
            print('no annotation')
        pass

    def put(self, basename, anno):
        try:
            annos = self.aprof['annos']
            panno = json.loads('{"text":"'+anno+'"}')
            annos[basename] = panno
            self.aprof['annos'] = annos
            with open(self.initialdir+"/."+AnnoSaveFile,"w",encoding="utf-8") as f:
                f.write(json.dumps(self.aprof))
        except:
            pass
        pass

    def get(self, basename):
        try:
            annos = self.aprof['annos']
            panno = annos[basename]
            return panno['text']
        except:
            pass
        return ""

class PageProp():
    def __init__(self, filename):
        
        pass
"""
class ScrCanvas():
    def __init__(self, bf, width, height, bg ):
        self.canvas = tkinter.Canvas(bf, width=width, height=height, bg=bg)
        self.canvas.bind('<Motion>', self.onMotion)

        self.onScr=False
        pass

    def onPress(self, event):
        print("onPress",event)
        self.scrSx = -1
        self.scrSy = -1
        self.onScr = True
        pass

    def onRelease(self, event):
        print("onRelease",event)
        self.scrSx = -1
        self.scrSy = -1
        self.onScr = False
        pass

    def onMotion(self,event):
        if self.onScr == False:
            return
        if( ax==0):
            return
        if( self.scrSx < 0):
            self.scrSx = event.x
            self.scrSy = event.y
            return

        movx, movy = self.scr2pic( self.scrSx - event.x , self.scrSy - event.y)
        self.clipSx, self.clipSy, self.clipEx, self.clipEy = self.correctClip(self.clipSx + movx, self.clipSy + movy, self.clipEx + movx,self.clipEy + movy)
        self.cimg = self.fimg[self.clipSy:self.clipEy, self.clipSx:self.clipEx]
        self.setImg(self.cimg)
        self.scrSx = event.x
        self.scrSy = event.y
        pass

    def setImg( self, img ):
        global ax,ay,xbias,ybias

        self.canvas.delete("all")
    
        try:
            vw, vh = self.getFitsize(img.shape[1], img.shape[0], Vwidth, Vheight)

            rimg = cv2.resize(img , (vw, vh))
            rgbimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgbimg)
            self.tkimg = ImageTk.PhotoImage(pil_image)
            self.c1id=self.canvas.create_image(Vwidth/2, Vheight/2, image=self.tkimg)

            if img.shape[1] > img.shape[0]:
                ax = Vwidth / img.shape[1]
                ay = vh / img.shape[0]
                ybias = (Vheight - vh )/2
                xbias=0
            else:
                ax = vw / img.shape[1]
                ay = Vheight / img.shape[0]
                xbias = (Vwidth - vw )/2
                ybias = 0

            if self.prgEx > -1:
                sx, sy = self.fpos2spos(self.prgSx,self.prgSy)
                ex, ey = self.fpos2spos(self.prgEx,self.prgEy)
                self.canvas.create_rectangle(sx, sy, ex, ey, outline='blue' )

        except:
            self.canvas.create_text(75, 75, text = self.filename)
 

    def canvas(self):
        return self.canvas;
"""
class CAPapp():
    def __init__(self, **kwargs):
        global Vheight, Vwidth
        # ルート窓
        self.root = tkinter.Tk()
        print(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        Vwidth = self.root.winfo_screenwidth()
        Vheight = self.root.winfo_screenheight() - 160

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0,weight=1)
        # フレーム
        self.frame=ttk.Frame(self.root,padding=10)
        self.frame.columnconfigure(0,weight=1)
        self.frame.rowconfigure(0,weight=1)
        self.frame.grid(sticky=(tkinter.N,tkinter.W,tkinter.S,tkinter.E))
        # ボタンバー
        self.fbar=ttk.Frame(self.frame,padding=4)
        self.fbar.columnconfigure(0,weight=1)
        self.fbar.rowconfigure(0,weight=1)
        self.fbar.grid(sticky=(tkinter.N,tkinter.W,tkinter.S,tkinter.E))

        self.barlabel = tkinter.StringVar()
        self.barlabel.set("filename")
        self.label = ttk.Label(self.fbar, textvariable=self.barlabel)
        self.label.grid(row=0,column=0)

        self.anolabelvar = tkinter.StringVar()
        self.anolabelvar.set("annotation")
        self.anolabel = ttk.Label(self.fbar, textvariable=self.anolabelvar)
        self.anolabel.grid(row=1,column=0)

        self.btnColumn = 1

        self.addBtn( "p", self.gopage, 6)
        self.addBtn( "Ann", self.add_annotation, 6)
        self.addBtn( "F", self.filter, 6)
        self.addBtn( "-", self.toshrink, 8)
        self.addBtn( "+", self.toenlarge, 8)
        self.addBtn( "↶", self.back_clicked, 6)
        self.addBtn( "←", self.before_clicked, 8)
        self.addBtn( "→", self.next_clicked, 8)
        self.addBtn( "Open", self.fileopen_clicked, 8)

        # Canvas
        self.canvas=tkinter.Canvas(self.root, width=Vwidth, height=Vheight, bg='white')
        self.canvas.grid(row=2,column=0)

        self.canvas.bind('<Double-1>', self.clickImg)
        self.canvas.bind('<ButtonPress-1>', self.onPress)
        self.canvas.bind('<ButtonRelease>', self.onRelease)
        self.canvas.bind('<Motion>', self.onMotion)

        self.img = []
        self.fimg = []
        self.scrSx = -1
        self.scrSy = -1

        self.initialdir='.'
        self.aprof = {}
        self.onScr = False

    def addBtn(self, title, func, btn_width):
        self.fileopenBtn = ttk.Button(
                self.fbar, text=title, width=btn_width,
                command=func
                )
        self.fileopenBtn.grid(row=0,column=self.btnColumn)
        self.btnColumn = self.btnColumn + 1
        pass

    def toenlarge(self):
        self.clipSx, self.clipSy, self.clipEx, self.clipEy = self.enlargement(self.clipSx, self.clipSy, self.clipEx, self.clipEy)
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.setImg(self.cimg)

    def toshrink(self):
        self.clipSx, self.clipSy, self.clipEx, self.clipEy = self.shrink(self.clipSx, self.clipSy, self.clipEx, self.clipEy)
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.setImg(self.cimg)

    def sliceScanX(self, px, py, csy, cey):

        csy, cey = self.scanareaY(py)
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

        return minx, maxx
    
    def sliceScanY(self, px, py, csx, cex):

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

        return miny, maxy
    
    def onPress(self, event):
        print("onPress",event)
        self.scrSx = -1
        self.scrSy = -1
        self.onScr = True
        pass

    def onRelease(self, event):
        print("onRelease",event)
        self.scrSx = -1
        self.scrSy = -1
        self.onScr = False
        pass

    def onMotion(self,event):
        if self.onScr == False:
            return
        print("onMotion ",event)
        #if( ax==0):
        #    return
        if( self.scrSx < 0):
            self.scrSx = event.x
            self.scrSy = event.y
            return

        movx, movy = self.scr2pic( self.scrSx - event.x , self.scrSy - event.y)

        self.clipSx, self.clipSy, self.clipEx, self.clipEy = self.correctClip(self.clipSx + movx, self.clipSy + movy, self.clipEx + movx,self.clipEy + movy)

        self.cimg = self.fimg[self.clipSy:self.clipEy, self.clipSx:self.clipEx]

        self.setImg(self.cimg)

        self.scrSx = event.x
        self.scrSy = event.y

        pass

    def correctClip(self, minx, miny, maxx, maxy):
        w = maxx - minx
        if( w > self.img.shape[1]):
            minx = 0
            maxx = self.img.shape[1]
        h = maxy - miny
        if( h > self.img.shape[0]):
            miny = 0
            maxy = self.img.shape[0]

        if Vwidth > Vheight:
            tw = (Vwidth / Vheight) * h
            if tw > (maxx-minx):
                ta = int((tw - (maxx-minx))/2)
                minx = minx - ta
                if minx < 0:
                    minx = 0
                maxx = maxx + ta
                if maxx > self.img.shape[1]:
                    maxx = self.img.shape[1]
        
        if( miny < 0):
            miny = 0
            maxy = miny + h
        if( maxy > self.img.shape[0]):
            maxy = self.img.shape[0]
            miny = maxy - h
        if( minx < 0):
            minx = 0
            maxx = minx + w
        if( maxx > self.img.shape[1]):
            maxx = self.img.shape[1]
            minx = maxx - w
        
        return minx, miny, maxx, maxy

    def clickImg(self, event):
        px, py = self.scr2pic(event.x,event.y)

        ori = self.localscan(px,py)
        tx,ty = event.x,event.y

        #self.canvas.create_rectangle(tx-4,ty-4,tx+4,ty+4,outline="red")

        if ori=="H":
            csy, cey = self.scanareaY(py)
            minx, maxx = self.sliceScanX(px,py,csy,cey)            
            
            sx, sy=self.pic2scr(minx,csy)
            ex, ey=self.pic2scr(maxx,cey)
        
            spfrm = int(splen *1.5)
            miny = py
            headx = int(((maxx-minx)/4)+minx)
            for y in range(py,0,-spfrm):
                limg = self.fimg[y:y+spfrm,minx:headx]
                limstd = np.std(limg)
                miny = y
                if limstd < thr:
                    break

            for y in range(py,self.img.shape[0],spfrm):
                limg = self.fimg[y:y+spfrm,minx:headx]
                limstd = np.std(limg)
                maxy = y
                if limstd < thr:
                    break

            maxy = maxy + splen
            sx, sy=self.pic2scr(minx,miny)
            ex, ey=self.pic2scr(maxx,maxy)
            #self.canvas.create_rectangle(sx, sy, ex, ey, outline='blue' )
            #self.canvas.create_line(tx, ty, sx, sy, fill='blue' )

            Vw = maxx - minx
            Vh = ( Vw * Vheight)/Vwidth
            maxy = int(miny + Vh)

        if ori=="V": #縦書き
            csx, cex = self.scanareaX(px)
            miny, maxy = self.sliceScanY(px,py,csx,cex) 
            print("miny,maxy:", miny, maxy)
            
            sx, sy=self.pic2scr(csx,miny)
            ex, ey=self.pic2scr(cex,maxy)
            #self.canvas.create_rectangle(sx, sy, ex, ey, outline='red' )
        
            spfrm = int(splen *1.5)
            minx = px
            heady = int(((maxy-miny)/4)+miny)
            for x in range(px,0,-spfrm):
                limg = self.fimg[miny:heady,x:x+spfrm]
                limstd = np.std(limg)
                minx = x
                if limstd < thr:
                    break

            for x in range(px,self.img.shape[1],spfrm):
                limg = self.fimg[miny:heady,x:x+spfrm]
                limstd = np.std(limg)
                maxx = x
                if limstd < thr:
                    break

            maxx = maxx + splen
            sx, sy=self.pic2scr(minx,miny)
            ex, ey=self.pic2scr(maxx,maxy)
            #self.canvas.create_rectangle(sx, sy, ex, ey, outline='blue' )
            #self.canvas.create_line(tx, ty, ex, sy, fill='blue' )
            
            Vh = maxy - miny
            Vw = ( Vh * Vwidth)/ Vheight
            minx = int(maxx - Vw)
            
        self.clipSx, self.clipSy, self.clipEx, self.clipEy = self.shrink(minx, miny, maxx, maxy)
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.setImg(self.cimg)
                       
    def shrink(self, minx, miny, maxx, maxy):
        dw = int((maxx - minx) * 0.02)
        dh = int((maxy - miny) * 0.02)
        return self.correctClip(minx-dw,miny-dh,maxx+dw,maxy+dh)

    def enlargement(self, minx, miny, maxx, maxy):
        dw = int((maxx - minx) * 0.02)
        dh = int((maxy - miny) * 0.02)
        return self.correctClip(minx+dw,miny+dh,maxx-dw,maxy-dh)

    def getFitsize(self, w, h, sw, sh ):
        if sw > sh:
            vh = sh
            vw = w * (vh/h)
        else:
            vw = sw
            vh = h * (vw/w)
        return int(vw), int(vh)
            
    def setImg( self, img ):
        global ax,ay,xbias,ybias

        self.canvas.delete("all")
    
        try:
            vw, vh = self.getFitsize(img.shape[1], img.shape[0], Vwidth, Vheight)

            rgbimg = cv2.resize(img , (vw, vh))

            #rgbimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgbimg)
            self.tkimg = ImageTk.PhotoImage(pil_image)
            self.c1id=self.canvas.create_image(Vwidth/2, Vheight/2, image=self.tkimg)

            if img.shape[1] > img.shape[0]:
                ax = Vwidth / img.shape[1]
                ay = vh / img.shape[0]
                ybias = (Vheight - vh )/2
                xbias=0
            else:
                ax = vw / img.shape[1]
                ay = Vheight / img.shape[0]
                xbias = (Vwidth - vw )/2
                ybias = 0

        except:
            self.canvas.create_text(75, 75, text = self.filename)

    def filter0(self):
        filimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2GRAY)
        filimg = cv2.convertScaleAbs(filimg,alpha = 3,beta = -200 )
        self.setImg(filimg)

    def filter(self):
        self.fimg = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        outimage00 = BookSharpener.sharpenMem( self.fimg, 5.0, 64, 0 )
        outimage32 = BookSharpener.sharpenMem( self.fimg, 5.0, 64, 32 )
        self.fimg = cv2.addWeighted(outimage00, 0.5, outimage32, 0.5, 0)
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.setImg(self.cimg)

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

        return ori

    def fpos2spos(self,x,y):
        sx = int((x-self.clipSx) * Vwidth / (self.clipEx - self.clipSx))
        sy = int((y-self.clipSy) * Vheight / (self.clipEy - self.clipSy))
        return sx,sy

    def fpos2sposSub(self,x,y):
        sx = int((x-self.clipSx) * Vwidth / (self.clipEx - self.clipSx))
        sy = int((y-self.clipSy) * Vheight / (self.clipEy - self.clipSy))
        return sx,sy


    def pic2scr(self, px, py):
        return  px*ax+xbias, py*ay+ybias

    def scr2pic(self, sx, sy):
        return  int((sx-xbias)/ax), int((sy-ybias)/ay)

    def load_annotation(self):
        self.aprof = json.loads('{"annos":{".annofilerx.json":{"text":"annotation"}}}')
        try:
            with open(self.initialdir+'/.annofilerx.json','r',encoding='utf-8') as f:
                self.aprof = json.load(f)
                annos = self.aprof['annos']
                for a in annos:
                    print(a,annos[a])
        except:
            print('no annotation')

            
        pass

    def add_annotation(self):
        istr = self.anolabelvar.get()
        anno = simpledialog.askstring(title="",prompt="annotation",initialvalue=istr)
        if anno!= None:
            self.annos.put(os.path.basename(self.filename),anno)
            self.anolabelvar.set( anno )

    def gopage(self):
        pagestr = simpledialog.askstring(title="go",prompt="go\npage")
        if pagestr!= None:
            p = int(pagestr)
            if p >= 0 and p < len(self.pfiles):
                self.filename =  self.initialdir+"/"+self.pfiles[p]
                self.filereload()

    def fileopen_clicked(self):
        filename = filedialog.askopenfilename(initialdir=self.initialdir)
        if filename:
            self.annos = AnnoFiler(filename)
            self.initialdir= os.path.dirname(filename)
            self.pfiles = os.listdir(self.initialdir)
            self.pfiles.sort()
            if(os.path.basename(filename)==CpageSaveFile
                or os.path.basename(filename)==AnnoSaveFile ):
                try:
                    with open(self.initialdir+'/'+CpageSaveFile,'r') as f:
                        prof = json.load(f)
                        filename = self.initialdir+"/"+prof['lastfile']
                except:
                    pass

            self.filename =  filename
            self.filereload()
            print( self.filename, self.img.shape[1], self.img.shape[0] )

    def filereload(self):
        self.canvas.delete("all")
        try:
            with open( self.filename, 'rb') as f:
                basename=os.path.basename(self.filename)
                p=self.pfiles.index(basename)
                self.barlabel.set(str(p)+"/"+str(len(self.pfiles))+" "+self.filename)
                atext = self.annos.get(basename)
                self.anolabelvar.set( atext )
                fdata =f.read()
                inp = np.frombuffer(fdata, dtype = 'int8')
                self.img = cv2.imdecode(inp, cv2.IMREAD_UNCHANGED)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                self.fimg = cv2.imdecode(inp, cv2.IMREAD_UNCHANGED)
                self.fimg = cv2.cvtColor(self.fimg, cv2.COLOR_BGR2RGB)
                self.cimg = self.img
                self.clipSx =0
                self.clipSy =0
                self.clipEx = self.img.shape[1]
                self.clipEy = self.img.shape[0]
                self.setImg(self.img)
                with open(self.initialdir+'/'+CpageSaveFile,"w") as f:
                    f.write('{"lastfile":"'+basename+'"}')
        except:
            self.canvas.create_text(75, 75, text = self.filename)
            pass

    def next_clicked(self):
        filename = os.path.basename(self.filename)
        p = self.pfiles.index(filename)
        print(len(self.pfiles),p,self.pfiles[p])
        p = p + 1
        if p >= len(self.pfiles):
            p = len(self.pfiles) - 1
        self.filename =  self.initialdir+"/"+self.pfiles[p]
        self.filereload()
        pass

    def before_clicked(self):
        filename = os.path.basename(self.filename)
        p = self.pfiles.index(filename)
        print(len(self.pfiles),p,self.pfiles[p])
        p = p - 1
        if p <= 0:
            p = 0
        self.filename =  self.initialdir+"/"+self.pfiles[p]
        self.filereload()
        pass

    def back_clicked(self):
        self.filereload()

    def run(self):
        self.root.mainloop()

    def rectangle( self, sx, sy, ex, ey, col ):
        self.canvas.create_rectangle(int((sx)*ax+xbias), int((sy)*ay+ybias), int(ex*ax+xbias), int(ey*ay+ybias), fill=col, )


app = CAPapp()
app.run()

