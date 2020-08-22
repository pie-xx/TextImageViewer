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
    Blocksize = 64
    Testimagefile = imgfile
    TestimageTitle = Testimagefile.split('.')[0]

    bookimg = cv2.imread( Testimagefile )
    img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)
#    for y in range( 0, img_gray.shape[0] ):
#        for x in range( 0, img_gray.shape[1] ):
#            img_gray[y][x] = 256 - img_gray[y][x]    

    print( "width", img_gray.shape[1], "height", img_gray.shape[0] )

    slim = getStdThrsh(img_gray, Blocksize)
    outimage32 = sharpenMem( img_gray, slim, Blocksize, 32 )
    cv2.imwrite("outimage32g.png", outimage32)
    
    img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)
    outimage00 = sharpenMem( img_gray, slim, Blocksize, 0 )
    cv2.imwrite("outimage00g.png", outimage00)

    for y in range( 0, img_gray.shape[0] ):
        for x in range( 0, img_gray.shape[1] ):
            if outimage00[y][x] > outimage32[y][x]:
                outimage00[y][x] = outimage32[y][x]
    
    #outimage = cv2.addWeighted(outimage00, 0.5, outimage32, 0.5, 0)

    rtn = getOutputName(TestimageTitle, slim)
    cv2.imwrite(rtn, outimage00 )
    return rtn

def sharpenMem(img_gray, slim, Blocksize, bias):
    Bbias = 0.2
    outimage = img_gray.copy()
    cannyimg = cv2.Canny(img_gray,64,128) 

    for y in range( bias, img_gray.shape[0], Blocksize ):
        s = ""
        for x in range( bias, img_gray.shape[1], Blocksize ):
            himg = cannyimg[y:y+Blocksize, x:x+Blocksize]
            avr = np.mean( himg )

            pimg = img_gray[y:y+Blocksize, x:x+Blocksize]
            #std = np.std( pimg )
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
            
            #if std < slim:
            if avr < 0.1:
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

    return outimage


if __name__ =='__main__':
    sharpenImg('tarama36p.jpg')

