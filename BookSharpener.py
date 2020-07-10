import cv2
from matplotlib import pyplot as plt
import numpy as np

Testimagefile = 'testimage.jpg'
Blocksize = 64

def getUmed(img):
    med = np.median(img) 
    fild = img[img < med]
    umed = np.median(fild) 
    #fild2 = fild[fild < umed]
    #umed = np.median(fild2) 
    #fild3 = fild2[fild2 < umed]
    #umed = np.median(fild3) 
    return( umed )

def sharpener( img_gray, slim, pname ):
    
    dst=img_gray.copy()
    for y in range(img_gray.shape[0]):
        for x in range(img_gray.shape[1]):
            if dst[y][x] > slim:
                dst[y][x] = 255
            else:
                dst[y][x] = dst[y][x] / 2

    cv2.imwrite(pname+'{:03d}.jpg'.format(slim), dst )

bookimg = cv2.imread( Testimagefile )
img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)
"""
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(Blocksize,Blocksize))
cl1 = clahe.apply(img_gray)
cv2.imwrite('testimage_cv.jpg', cl1 )

"""
imgHeight = int(img_gray.shape[0])
imgWidth = int(img_gray.shape[1])

for y in range( 0, imgHeight, Blocksize ):
    for x in range( 0, imgWidth, Blocksize ):
        pimg = img_gray[y:y+Blocksize, x:x+Blocksize]
        std = np.std( pimg )
        minv = np.min( pimg )
        maxv = np.max( pimg )
        pimg -= minv

        slim = getUmed( pimg )
        print( y, x, slim, std, minv, maxv )
        if std < 6.0:
            for sy in range (pimg.shape[0]):
                for sx in range( pimg.shape[1] ):
                    img_gray[y+sy][x+sx] = 255
        else:
            for sy in range (pimg.shape[0]):
                for sx in range( pimg.shape[1] ):
                    if maxv != minv:
                        img_gray[y+sy][x+sx] = int((img_gray[y+sy][x+sx] *255.0)/(maxv - minv))                    
                    if pimg[sy][sx] > slim:
                    #    img_gray[y+sy][x+sx] = 256 - (256 - img_gray[y+sy][x+sx]) / 2
                        v = img_gray[y+sy][x+sx]
                        v = v * 2
                        if v > 256:
                            v = 255
                        img_gray[y+sy][x+sx] = v
                    else:
                        img_gray[y+sy][x+sx] = pimg[sy][sx] / 3
     
cv2.imwrite('testimage_f.jpg', img_gray )
cv2.imshow('testimage_f1', img_gray )

