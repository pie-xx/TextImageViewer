#from numba import jit
import cv2
from matplotlib import pyplot as plt
import numpy as np

def getStdThrsh(img, Blocksize):
    stds = []
    for y in range( 0, img.shape[0], Blocksize ):
        for x in range( 0, img.shape[1], Blocksize ):
            pimg = img[y:y+Blocksize, x:x+Blocksize]
            std = np.std( pimg )
            minv = np.min( pimg )
            maxv = np.max( pimg )
            stds.append(std)

    hist = np.histogram( stds, bins=64 )
    peaki = np.argmax(hist[0])   

    #plt.hist( stds, bins=64 )
    #plt.show()

    slim = 6.0
    for n in range(peaki,len(hist[0])-1):
        if hist[0][n] < hist[0][n+1]:
            slim = hist[1][n+1]
            break

    #if slim > 6.0:
    #    slim = 6.0
    
    return slim

def getBWThrsh(img):
    med = np.median(img)
    fild = img[img < med]
    return np.median(fild)

def getWbias( img, bwthr ):
    wimg = img[ img > bwthr ]
    hist = np.histogram( wimg, bins=16 )
    agm = np.argmax(hist[0])
    return hist[1][agm]

def getOutputName( title, slim ):
    return title + "_s{:04.2f}.jpg".format( slim )

def sharpenImg(imgfile):
    Blocksize = 64
    Testimagefile = imgfile
    TestimageTitle = Testimagefile.split('.')[0]

    with open(Testimagefile, mode='rb') as f:
        fdata =f.read()
        inp = np.frombuffer(fdata, dtype = 'int8')
        bookimg = cv2.imdecode(inp, cv2.IMREAD_UNCHANGED)
    #bookimg = cv2.imread( Testimagefile )
    img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)

    print( "width", img_gray.shape[1], "height", img_gray.shape[0] )

    slim = getStdThrsh(img_gray, Blocksize)
    outimage32 = sharpenMem( img_gray, slim, Blocksize, 32 )
    cv2.imwrite("outimage32g.png", outimage32)
    
    img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)
    outimage00 = sharpenMem( img_gray, slim, Blocksize, 0 )
    cv2.imwrite("outimage00g.png", outimage00)

    #for y in range( Blocksize, img_gray.shape[0]-Blocksize*2 ):
    #    for x in range( 0, img_gray.shape[1] ):
    #        if outimage00[y][x] > outimage32[y][x]:
    #            outimage00[y][x] = outimage32[y][x]
    outimage = cv2.addWeighted(outimage00, 0.5, outimage32, 0.5, 0)

    rtn = getOutputName(TestimageTitle, slim)
    with open(rtn,mode='wb') as f:
        fv, bookimg = cv2.imencode(rtn,outimage)
        f.write(bookimg)

    #cv2.imwrite(rtn, outimage )
    return rtn

def neardevi(imgbl):
    d = 0.0
    for y in range(imgbl.shape[0]):
        for x in range(1, imgbl.shape[1] ):
            d = d + (imgbl[y][x-1]-imgbl[y][x])*(imgbl[y][x-1]-imgbl[y][x])
    d = d / (imgbl.shape[0]*imgbl.shape[1])
    return d        

#@jit
def sharpenMem(img_gray, slim, Blocksize, bias):
    Bbias = 0.2
    outimage = img_gray.copy()
    bimgl = np.zeros([Blocksize,int(bias)]) + 255
    bimgu = np.zeros([int(Blocksize/2),img_gray.shape[1]]) + 255
    yimgs=[]
    if bias!=0:
        yimgs.append(bimgu)

    for y in range( bias, img_gray.shape[0]-Blocksize, Blocksize ):
        s = ""

        ximgs=[]
        if bias!=0:
            ximgs.append(bimgl)
        for x in range( bias, img_gray.shape[1], Blocksize ):
            pimg = img_gray[y:y+Blocksize, x:x+Blocksize]
            std = np.std( pimg )
                 
            if std < slim:
                s = s + "_"
                ximg=np.zeros(pimg.shape) + 255
            else:
                s = s + "#"
                lut = np.zeros(256)
                white = int(np.median(pimg))
                black = int(white / 2)
                cnt = int(white - black)
                for n in range(cnt):
                    lut[black+n]=( int(256 * n / cnt) )
                for n in range(white,256):
                    lut[n]=(255)
                ximg=cv2.LUT(pimg,lut)

            ximgs.append(ximg)

        #if bias!=0:
        #    ximgs.append(bimgr)

        ximgsall=cv2.hconcat( ximgs )
        yimgs.append( ximgsall )
        print( "{:4d} {:s}".format( y, s ) )

    if bias==0:
        yimgs.append(bimgu)

    outimage = cv2.vconcat(yimgs)

    return outimage

if __name__ =='__main__':
    sharpenImg('city.jpg')

