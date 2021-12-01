#横または縦のライン方向についてスキャンした濃淡グラフ
# 画像を選択表示する。表示画像をクリックすると近傍64dot四方のヒストグラムを表示
#
import os
from posixpath import basename
import cv2
import tkinter
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import copy

from sklearn import cluster, preprocessing
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from numpy.lib.function_base import append
import json
from tkinter import simpledialog

#import pyocr
#import pyocr.builders
#from sklearn.cluster import KMeans

Vwidth = 1200
Vheight = 800
VSubwidth = 400
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

class CAPapp():
    def __init__(self, **kwargs):
        # ルート窓
        self.root = tkinter.Tk()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0,weight=1)
        # フレーム
        self.frame=ttk.Frame(self.root,padding=10)
        self.frame.columnconfigure(0,weight=1)
        self.frame.rowconfigure(0,weight=1)
        self.frame.grid(sticky=(tkinter.N,tkinter.W,tkinter.S,tkinter.E))
        # ボタンバー
        self.fbar=ttk.Frame(self.frame,padding=4)
        self.fbar.columnconfigure(10,weight=1)
        self.fbar.rowconfigure(0,weight=1)
        self.fbar.grid(sticky=(tkinter.N,tkinter.W,tkinter.S,tkinter.E))
        #self.fbar.grid(row=0,column=0)
        # ボタンバー
        self.cvfram=ttk.Frame(self.frame,padding=4)
        self.cvfram.columnconfigure(0,weight=1)
        self.cvfram.rowconfigure(0,weight=1)
        self.cvfram.grid(row=0,column=0)

        self.barlabel = tkinter.StringVar()
        self.barlabel.set("filename")
        self.label = ttk.Label(self.fbar, textvariable=self.barlabel)
        self.label.grid(row=0,column=0)

        self.anolabelvar = tkinter.StringVar()
        self.anolabelvar.set("annotation")
        self.anolabel = ttk.Label(self.fbar, textvariable=self.anolabelvar)
        self.anolabel.grid(row=1,column=0)

        self.btnColumn = 1
        self.buttons = []

        self.addBtn( "A", self.add_annotation, 6)
        self.addBtn( "Slice", self.searchTextLine, 6)
        self.addBtn( "F●", self.contrastLow, 6)
        self.addBtn( "F", self.contrastMid, 6)
        self.addBtn( "F○", self.contrastHigh, 6)
        self.addBtn( "-", self.toshrink, 8)
        self.addBtn( "+", self.toenlarge, 8)
        self.addBtn( "↶", self.back_clicked, 6)
        self.addBtn( "←", self.before_clicked, 8)
        self.addBtn( "→", self.next_clicked, 8)
        self.addBtn( "Open", self.fileopen_clicked, 8)

        # Canvas
        self.canvas=tkinter.Canvas(self.frame, width=Vwidth, height=Vheight, bg='white')
        self.canvas.grid(row=2,column=0)
        # Canvas Sub
        self.subcanvas=tkinter.Canvas(self.frame, width=VSubwidth, height=Vheight, bg='white')
        self.subcanvas.grid(row=2,column=1)

        self.canvas.bind('<Double-1>', self.clickImg)
        self.canvas.bind('<ButtonPress-1>', self.onPress)
        self.canvas.bind('<ButtonRelease>', self.onRelease)
        self.canvas.bind('<Motion>', self.onMotion)

        self.img = []
        self.fimg = []
        self.scrSx = -1
        self.scrSy = -1

        self.initialdir='.'

        self.onScr = False

        self.beta = 0
        self.prgEx = -1
        self.ori = ""

    def addBtn(self, title, func, btn_width):

        btn = ttk.Button(
                self.fbar, text=title, width=btn_width,
                command=func
                )
        btn.grid(row=0,column=self.btnColumn)
        self.buttons.append(btn)
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

    def correctClip(self, minx, miny, maxx, maxy):
        w = maxx - minx
        if( w > self.img.shape[1]):
            minx = 0
            maxx = self.img.shape[1]

        h = maxy - miny
        if( h > self.img.shape[0]):
            miny = 0
            maxy = self.img.shape[0]

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

        self.ori = self.localscan(px,py)

        if self.ori=="H":
            csy, cey = self.scanareaY(py)
            minx, maxx = self.sliceScanX(px,py,csy,cey)            
               
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

            Vw = maxx - minx
            Vh = ( Vw * Vheight)/Vwidth
            maxy = int(miny + Vh)

        if self.ori=="V": #縦書き
            csx, cex = self.scanareaX(px)
            miny, maxy = self.sliceScanY(px,py,csx,cex) 
            print("miny,maxy:", miny, maxy)
            
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

            Vh = maxy - miny
            Vw = ( Vh * Vwidth)/ Vheight
            minx = int(maxx - Vw)
            
        self.clipSx, self.clipSy, self.clipEx, self.clipEy = self.shrink(minx, miny, maxx, maxy)
        
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.prgSx,self.prgSy,self.prgEx,self.prgEy = minx, miny, maxx, maxy
        
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
        if w < h:
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
        
    def searchTextLine(self):
        global splen
        clength = []
        connects = []
        if self.ori == "H":
            chlines=[]
            for x in range( self.prgSx, self.prgEx, splen):
                sx, sy = self.fpos2spos(x,self.prgSy)
                ex, ey = self.fpos2spos(x,self.prgEy)
                #self.canvas.create_line(sx, sy, ex, ey, fill='white' )
                inw = 0
                charry = []
                chst = 0
                for y in range(self.prgSy,self.prgEy,8):
                    limg = self.fimg[y:y+8,x:x+splen*2]
                    std = np.std(limg)
                    if std > thr:
                        if inw==0: #文字開始
                            chst = y
                            pass                        
                        sx, sy = self.fpos2spos(x,y)
                        ex, ey = self.fpos2spos(x+8,y+8)
                        #self.canvas.create_rectangle(sx, sy, ex, ey, fill='green' )
                        inw = 1
                    else:
                        if inw==1: #文字終了
                            if( y - chst) > splen*1.5:
                                d = int((y - chst )/2)
                                charry.append((chst,chst+d))
                                clength.append(d)
                                chst = chst + d
                            if len(charry)==0:
                                charry.append((chst,y))
                                clength.append(y-chst)
                            if y -charry[-1][0] < splen*1.5:
                                #print("short",charry[0],(chst,x))
                                charry[-1] = (charry[-1][0],y)
                            else:
                                charry.append((chst,y))                            
                            pass
                        inw = 0
                if inw==1: #文字終了
                    charry.append((chst,y))
                    clength.append(y-chst)
                chlines.append(charry)

            for x in range(1,len(chlines)):
                xline = chlines[x]
                for y in range(len(xline)):
                    bxline = chlines[x-1]
                    for b in range(len(bxline)):
                        bp = bxline[b]
                        if self.isConnect(xline[y],bp):
                            #print(y,x,xline[y],bp)
                            xpos = self.prgSx + x*splen
                            #self.drawline(xpos-splen,(bp[0]+bp[1])/2,xpos,(xline[y][0]+xline[y][1])/2, width=8)
                            connects.append((
                                (xpos-splen, (bp[0]+bp[1])/2),
                                (xpos, (xline[y][0]+xline[y][1])/2)
                                ))


            cnt = len(connects)
            print( "connects len = ", cnt )
            tlines=[]
            lc=(0,0)
            for cn in range(1,int(len(connects))):
                cnct = connects[cn]
                e=False
                for t in tlines:
                    if t[-1]==cnct[0]:
                        t.append(cnct[1])
                        e = True
                        lc=cnct[0]
                        break
                if not e and lc!=cnct[0]:
                    newt = [cnct[0],cnct[1]]
                    tlines.append(newt)

            n=0
            lininx = []
            for t in tlines:
                lininx.append( (t, n) )
                n=n+1

            lininx.sort(key = lambda x: int(x[0][0][1]/splen*2)*1000+int(x[0][0][0]/splen*2), reverse=False)
            self.subcanvas.delete("all")
            sph = int(splen)
            n=0
            y=0
            x=0
            cnctxnum = int(VSubwidth/splen)
            self.cnctImg = self.fimg[ 0:4, 0:cnctxnum*splen]
            lineimgs = []
            for s in lininx:
                t=s[0]
                #self.drawline(t[0][0],t[0][1],t[-1][0],t[-1][1], fill="blue")
                #self.drawline(t[0][0]-sph,t[0][1]-sph,t[-1][0]+sph,t[-1][1]-sph, fill="blue")
                #self.drawline(t[0][0]-sph,t[0][1]+sph,t[-1][0]+sph,t[-1][1]+sph, fill="blue")
                #rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for m in range(1,len(t)):
                    #self.drawline(t[m-1][0],t[m-1][1]-sph,t[m][0],t[m][1]-sph, fill="blue")
                    self.drawline(t[m-1][0],t[m-1][1]+sph,t[m][0],t[m][1]+sph, fill="blue")
                    """
                    src_pts = np.array([[t[m-1][0],t[m-1][1]-sph], [t[m][0],t[m][1]-sph], 
                                        [t[m-1][0],t[m-1][1]+sph], [t[m][0],t[m][1]+sph]], dtype=np.float32)
                    dst_pts = np.array([[0, 0], [splen, 0], [splen, splen*2], [0, splen*2]], dtype=np.float32)
                    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    _img = cv2.warpPerspective(self.fimg, mat, (splen, splen*2))
                    """
                    px=int(t[m-1][0])
                    py=int(t[m-1][1])
                    _img = self.fimg[ py-sph:py+sph, px:px+sph]
                    lineimgs.append(_img)
                    x=x+splen
                    if( x > (VSubwidth - splen)):
                        y=y+splen*2
                        himg = cv2.hconcat(lineimgs)
                        if(himg.shape[1]==self.cnctImg.shape[1]):
                            self.cnctImg = cv2.vconcat([self.cnctImg, himg])
                        lineimgs=[]
                        x=0


                y=y+splen*2
                self.drawline(t[0][0]-sph,t[0][1]-sph,t[0][0]-sph,t[0][1]+sph, fill="blue")
                self.drawline(t[-1][0]+sph,t[-1][1]-sph,t[-1][0]+sph,t[-1][1]+sph, fill="blue")
                self.drawRectangle(t[0][0]-8,t[0][1]-8,t[0][0]+8,t[0][1]+8, fill="white")
                self.drawText(t[0][0],t[0][1],"{:d}".format(n))
                n=n+1
            cv2.imshow("ori",self.cnctImg)

        if self.ori == "V":
            chlines=[]
            for y in range( self.prgSy, self.prgEy, splen):
                sx, sy = self.fpos2spos(self.prgSx,y)
                ex, ey = self.fpos2spos(self.prgEx,y)
                #self.canvas.create_line(sx, sy, ex, ey, fill='white' )
                inw = 0
                charry = []
                chst = 0
                for x in range(self.prgSx,self.prgEx,8):
                    limg = self.fimg[y:y+splen,x:x+16]
                    std = np.std(limg)
                    if std > thr:
                        if inw==0: #文字開始
                            chst = x
                            pass
                        #self.drawRectangle(x,y,x+8,y+8)
                        inw = 1
                    else:
                        if inw==1: #文字終了
                            if( x - chst) > splen*1.8:
                                d = int((x - chst )/2)
                                charry.append((chst+d,x))
                                clength.append(d)
                                x = chst + d
                            if len(charry)==0:
                                charry.insert(0,(chst,x))
                                clength.append(x-chst)
                            if x-charry[0][0] < splen*1.5:
                                #print("short",charry[0],(chst,x))
                                charry[0] = (charry[0][0],x)
                            else:
                                charry.insert(0,(chst,x))
                                

                            pass
                        inw = 0
                if inw==1: #文字終了
                    charry.insert(0,(chst,x))
                    clength.append(x-chst)
                chlines.append(charry)


            connects.append( ((0,0),(0,0)) )
            for y in range(1,len(chlines)):
                yline = chlines[y]
                for x in range(len(yline)):
                    byline = chlines[y-1]
                    for b in range(len(byline)):
                        bp = byline[b]
                        if self.isConnect(yline[x],bp):
                            #print(y,x,yline[x],b,bp)
                            ypos = self.prgSy + y*splen
                            #self.drawline(bp[0],ypos-splen,yline[x][0],ypos)
                            #self.drawline(bp[1],ypos-splen,yline[x][1],ypos)
                            #self.drawline((bp[0]+bp[1])/2,ypos-splen,(yline[x][0]+yline[x][1])/2,ypos, width=8)
                            #if connects[-1][0]!=((bp[0]+bp[1])/2,ypos-splen):
                            #print( connects[-1] )
                            connects.append((
                                ((bp[0]+bp[1])/2, ypos-splen),
                                ((yline[x][0]+yline[x][1])/2, ypos)
                            ))
                            break

            cnt = len(connects)
            print( "connects len = ", cnt )
            tlines=[]
            lc=(0,0)
            for cn in range(1,int(len(connects))):
                cnct = connects[cn]
                e=False
                for t in tlines:
                    if t[-1]==cnct[0]:
                        t.append(cnct[1])
                        e = True
                        lc=cnct[0]
                        break
                if not e and lc!=cnct[0]:
                    newt = [cnct[0],cnct[1]]
                    tlines.append(newt)

            sph = int(splen)
            n=0

            lininx = []
            for t in tlines:
                lininx.append( (t, n) )
                n=n+1

            lininx.sort(key = lambda x: x[0][0], reverse=True)

            n=0
            for s in lininx:
                t=s[0]
                #self.drawline(t[0][0],t[0][1],t[-1][0],t[-1][1], fill="blue")
                self.drawline(t[0][0]-sph,t[0][1]-0,t[-1][0]-sph,t[-1][1]+sph, fill="blue")
                self.drawline(t[0][0]+sph,t[0][1]-0,t[-1][0]+sph,t[-1][1]+sph, fill="blue")
                self.drawline(t[0][0]-sph,t[0][1]-0,t[0][0]+sph,t[0][1]-0, fill="blue")
                self.drawline(t[-1][0]-sph,t[-1][1]+sph,t[-1][0]+sph,t[-1][1]+sph, fill="blue")
                self.drawRectangle(t[0][0]-8,t[0][1]-8,t[0][0]+8,t[0][1]+8, fill="white")
                self.drawText(t[0][0],t[0][1],"{:d}".format(n))
                n=n+1

        #plt.hist(clength,bins=128)
        #plt.show()
        pass

    def drawSubImage(self, img, psx, psy, vw, vh, fname):
        #rimg = cv2.resize(img , (vw, vh))
        #cv2.imwrite( fname, img )
        #rgbimg = cv2.imread(fname)
        print(psx,psy)
        #rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #pil_image = Image.fromarray(img)
        #stkimg = ImageTk.PhotoImage(pil_image)

        #sx, sy = self.fpos2spos(psx,psy)
        self.c1id=self.subcanvas.create_image(psx, psy, image=img)
        self.c1id=self.subcanvas.create_rectangle(psx, psy, psx+8, psy+8)

    def drawSubRectangle(self,psx, psy, pex, pey, fill="red"):
        sx, sy = self.fpos2spos(psx,psy)
        ex, ey = self.fpos2spos(pex,pey)
        self.subcanvas.create_rectangle(psx, psy, pex, pey, fill=fill )

    def isNewTline(self,t,cnct):
        pass

    def drawText(self,x,y,text,col="black"):
        sx, sy = self.fpos2spos(x,y)
        self.canvas.create_text(sx, sy, text=text, fill=col)

    def drawline(self,psx, psy, pex, pey, fill="red", width=2):
        sx, sy = self.fpos2spos(psx,psy)
        ex, ey = self.fpos2spos(pex,pey)
        self.canvas.create_line(sx, sy, ex, ey, fill=fill, width=width )

    def drawRectangle(self,psx, psy, pex, pey, fill="red"):
        sx, sy = self.fpos2spos(psx,psy)
        ex, ey = self.fpos2spos(pex,pey)
        self.canvas.create_rectangle(sx, sy, ex, ey, fill=fill )

    def isConnect(self,p,q):
        pc = (p[0]+p[1])/2
        qc = (q[0]+q[1])/2
        if (q[0] <= pc and pc <= q[1] ) :
            return True
        if (p[0] <= qc and qc <= p[1] ) :
            return True
        return False

    def isConnectO(self,p,q):
        if (q[0] <= p[0] and p[0] <= q[1] ) or (q[0] <= p[1] and p[1] <= q[1] ):
            return True
        if (p[0] <= q[0] and q[0] <= p[1] ) or (p[0] <= q[1] and q[1] <= p[1] ):
            return True
        return False

    def filter2(self):
        fig, ax = plt.subplots(nrows=3, ncols=8, sharex=True, sharey=True)
        for y in range(3):
            for x in range(8):
                filimg = cv2.convertScaleAbs(self.cimg,alpha = y*0.5 + 1.5, beta = -25*x )
                ax[y,x].imshow(filimg)
                title = str(-25*x)
                ax[y,x].set_title(title)
        plt.show()
        pass

    def filter3(self):
        """
        filimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2GRAY)
        vars = []
        reso = 8
        for y in range(0,filimg.shape[0],reso):
            for x in range(0,filimg.shape[1],reso):
                limg = self.cimg[y:y+reso,x:x+reso]
                v = np.var(limg)
                vars.append(v)
                #if v > 100:
                #    cv2.rectangle(filimg, (x, y), (x+8,y+8), (0,0,255), 2 )

        self.setImg(filimg)

        #histup = np.histogram(vars, bins=50, range=None, normed=False, weights=None, density=None)
        #plt.plot(histup[1][1:], histup[0])
        #plt.show()

        dt = np.fromiter(vars, dtype=float)
        xdt = dt.reshape([dt.shape[0],1])

        xm_c = kmeans_plusplus_initializer(xdt, 2).initialize()
        xm_i = xmeans(data=xdt, initial_centers=xm_c, kmax=8, ccore=True)
        xm_i.process()
        classes = len(xm_i._xmeans__centers)
        predict = xm_i.predict(xdt)

        mins = []
        for i in range(classes):
            batch_predict = xdt[predict==i]
            print( i, np.min(batch_predict), np.max(batch_predict) )
            mins.append( np.max(batch_predict))
            n, _, _ = plt.hist(batch_predict, bins=100, alpha=0.5, label="class="+str(i))
            plt.vlines(xm_i._xmeans__centers[i], 0, max(n))

        minsp = np.min(mins)
        print(minsp)
        p=0
        for y in range(0,filimg.shape[0],reso):
            for x in range(0,filimg.shape[1],reso):
                if xdt[p] > minsp:
                    pass
                    #cv2.rectangle(filimg, (x, y), (x+8,y+8), (0,0,255), 2 )
                else:
                    cv2.rectangle(filimg, (x, y), (x+reso,y+reso), (255,255,255), thickness=-1 )
                p = p + 1
        """

        filimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2GRAY)
        histup = cv2.calcHist([filimg],[0],None,[256],[0,256]) 
        plt.plot(histup)
        filimg = cv2.convertScaleAbs(filimg,alpha = 3,beta = -50 )    
        #histup = cv2.calcHist([filimg],[0],None,[256],[0,256]) 
        #plt.plot(histup)
        self.setImg(filimg)
        plt.show()

    def contrast(self, beta):
        #self.fimg = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.fimg = self.fimg + beta
        self.fimg = self.fimg * 2.5 
        pass
    
    def contrastLow(self):
        #filimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2GRAY)
        #filimg = cv2.convertScaleAbs(filimg,alpha = 2.5,beta = -200 )
        #self.setImg(filimg)
        self.fimg = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.fimg = cv2.convertScaleAbs(self.fimg,alpha = 2.5,beta = -200 )
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.setImg(self.cimg)

    def contrastMid(self):
        self.fimg = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.fimg = cv2.convertScaleAbs(self.fimg,alpha = 2.5,beta = -125 )
        #self.contrast(-125)
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.setImg(self.cimg)

    def contrastHigh(self):
        #filimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2GRAY)
        #filimg = cv2.convertScaleAbs(filimg,alpha = 2.5,beta = -50 )
        #self.setImg(filimg)
        self.fimg = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.fimg = cv2.convertScaleAbs(self.fimg,alpha = 2.5,beta = -50 )
        self.cimg = self.fimg[self.clipSy:self.clipEy,self.clipSx:self.clipEx]
        self.setImg(self.cimg)

    def filter(self):
        """
        histup = cv2.calcHist([self.cimg],[0],None,[256],[0,256]) 
        maxv = -1
        vinx = -1
        inx = 0
        for v in histup:
            if v > maxv:
                maxv = v
                vinx = inx
            inx = inx + 1
        print( vinx, maxv )
        plt.plot(histup)
        """
        filimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2GRAY)
        filimg = cv2.convertScaleAbs(filimg,alpha = 2.5,beta = -125 )
        #print( 256/(256-vinx), vinx - 256  )
        """
        filimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2GRAY)
        histup = cv2.calcHist([filimg],[0],None,[256],[0,256]) 
        plt.plot(histup)
        plt.show()
        
        cv =[]
        ci =[]
        cimg = np.ravel( filimg)
        for n in range(1,256):
            print(n)
            upper = [x for x in cimg if x > n]
            lower = [x for x in cimg if x <= n]
            if( len(upper)*len(lower)!=0 ):
                us = np.std( upper )
                ls = np.std(lower)
                print(us,ls)
                cv.append( us + ls )
                ci.append( n )

        plt.plot(ci,cv)
        plt.show()
        """
        self.setImg(filimg)


        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #filimg = cv2.equalizeHist(filimg)
        #filimg = clahe.apply(filimg)
        #rimg = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)
        #self.oimg = cv2.imdecode(inp,cv2.IMREAD_UNCHANGED)

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

    def add_annotation(self):
        istr = self.anolabelvar.get()
        anno = simpledialog.askstring(title="",prompt="annotation",initialvalue=istr)
        if anno!= None:
            self.annos.put(os.path.basename(self.filename),anno)
            self.anolabelvar.set( anno )

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
        #self.prgEx = -1
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
                self.fimg = cv2.imdecode(inp, cv2.IMREAD_UNCHANGED)
                self.cimg = self.img
                self.clipSx =0
                self.clipSy =0
                self.clipEx = self.img.shape[1]
                self.clipEy = self.img.shape[0]
                self.setImg(self.fimg)
                with open(self.initialdir+'/'+CpageSaveFile,"w") as f:
                    f.write('{"lastfile":"'+basename+'"}')
        except:
            self.canvas.create_text(75, 75, text = self.filename)
            pass

    def next_clicked(self):
        filename = os.path.basename(self.filename)
        p = self.pfiles.index(filename)
        p = p + 1
        if p >= len(self.pfiles):
            p = len(self.pfiles) - 1
        self.filename =  self.initialdir+"/"+self.pfiles[p]
        self.filereload()
        pass

    def before_clicked(self):
        filename = os.path.basename(self.filename)
        p = self.pfiles.index(filename)
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

