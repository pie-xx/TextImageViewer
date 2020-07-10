import cv2
from matplotlib import pyplot as plt
import numpy as np

Testimagefile = 'city.jpg'
TestimageTitle = Testimagefile.split('.')[0]
Blocksize = 32
Slim = 5.0
Wbias = 1.5
Bbias = 0.3

def stdmap(img):
    stds = []
    for y in range( 0, img.shape[0], Blocksize ):
        for x in range( 0, img.shape[0], Blocksize ):
            pimg = img_gray[y:y+Blocksize, x:x+Blocksize]
            std = np.std( pimg )
            minv = np.min( pimg )
            maxv = np.max( pimg )
            stds.append(std)
    hist = np.histogram( stds, bins=64 )
    print( hist[0] )
    print(np.argmax(hist[0]))
    print( hist[1] )
    peaki = np.argmax(hist[0])
    if peaki == 0:
        peaki = 1
    for n in range(peaki,len(hist[0])):
        if hist[0][n-1] < hist[0][n]:
            Slim = hist[1][n+1]
            break
    print(Slim)
    plt.hist(stds, 100, [0, 100])
    plt.show()

def getUmed(img):
    med = np.median(img) 
    fild = img[img < med]
    umed = np.median(fild) 
    return( umed )

bookimg = cv2.imread( Testimagefile )
img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)

imgHeight = int(img_gray.shape[0])
imgWidth = int(img_gray.shape[1])

stdmap(img_gray)


for y in range( 0, imgHeight, Blocksize ):
    print( y )
    for x in range( 0, imgWidth, Blocksize ):
        pimg = img_gray[y:y+Blocksize, x:x+Blocksize]
        std = np.std( pimg )
        minv = np.min( pimg )
        maxv = np.max( pimg )
        avr = np.average(pimg)
        #pimg = cv2.equalizeHist(pimg)
        pimg -= minv

        wlim = getUmed( pimg )

        """
        for sy in range (pimg.shape[0]):
            for sx in range( pimg.shape[1] ):
                img_gray[y+sy][x+sx] = std*4
        """
        if std < Slim:
            for sy in range (pimg.shape[0]):
                for sx in range( pimg.shape[1] ):
                    img_gray[y+sy][x+sx] = 255
        else:
            for sy in range (pimg.shape[0]):
                for sx in range( pimg.shape[1] ):
                    if maxv != minv:
                        img_gray[y+sy][x+sx] = (img_gray[y+sy][x+sx]*255.0)/(maxv - minv)
                    if pimg[sy][sx] > wlim:
                        v = img_gray[y+sy][x+sx]
                        v = v * Wbias
                        if v > 255:
                            v = 255
                        img_gray[y+sy][x+sx] = v
                    else:
                        img_gray[y+sy][x+sx] = pimg[sy][sx] * Bbias

cv2.imwrite(TestimageTitle+'_mf152.jpg', img_gray )

