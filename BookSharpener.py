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

    if slim > 6.0:
        slim = 6.0
    
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
    Testimagefile = imgfile
    TestimageTitle = Testimagefile.split('.')[0]
    Blocksize = 64
    Bbias = 0.2

    bookimg = cv2.imread( Testimagefile )
    img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)
#    for y in range( 0, img_gray.shape[0] ):
#        for x in range( 0, img_gray.shape[1] ):
#            img_gray[y][x] = 256 - img_gray[y][x]    
    outimage = img_gray.copy()

    print( "width", img_gray.shape[1], "height", img_gray.shape[0] )

    slim = getStdThrsh(img_gray, Blocksize)
    for y in range( 0, img_gray.shape[0], Blocksize ):
        s = ""
        for x in range( 0, img_gray.shape[1], Blocksize ):
            pimg = img_gray[y:y+Blocksize, x:x+Blocksize]
            std = np.std( pimg )
            minv = np.min( pimg )
            maxv = np.max( pimg )
            pimg -= minv

            cimg = pimg.copy()
            if maxv != minv:
                for sy in range (cimg.shape[0]):
                    for sx in range( cimg.shape[1] ):
                        cimg[sy][sx] = (cimg[sy][sx]*255.0)/(maxv - minv)

            bwthrsh = getBWThrsh( pimg )
            wb = getWbias( cimg, bwthrsh )
            if wb == 0:
                wbias = 1.5
            else:
                wbias = 256 / wb
            
            if std < slim:
                s = s + "B"
                for sy in range (pimg.shape[0]):
                    for sx in range( pimg.shape[1] ):
                        outimage[y+sy][x+sx] = 255
            else:
                s = s + "_"
                for sy in range (cimg.shape[0]):
                    for sx in range( cimg.shape[1] ):
                        if cimg[sy][sx] > bwthrsh:
                            v = cimg[sy][sx] * wbias
                            if v > 255:
                                v = 255
                            outimage[y+sy][x+sx] = v
                        else:
                            outimage[y+sy][x+sx] = cimg[sy][sx] * Bbias
        print( "{:4d} {:s}".format( y, s ) )

    rtn = getOutputName(TestimageTitle, slim)
    cv2.imwrite(rtn, outimage )
    return rtn

if __name__ =='__main__':
    sharpenImg('tarama36p.jpg')

