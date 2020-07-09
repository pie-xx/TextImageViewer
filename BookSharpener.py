import cv2
from matplotlib import pyplot as plt
import numpy as np

def getUmed(img):
    med = np.median(img) 
    fild = img[img < med]
    umed = np.median(fild) 
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

bookimg = cv2.imread('20150429134907676.jpg')
img_gray = cv2.cvtColor(bookimg, cv2.COLOR_BGR2GRAY)

ha = int(img_gray.shape[0])
hh = int(img_gray.shape[0]/2)
orgupimg = img_gray[0:hh, 0:img_gray.shape[1]]
orgdwnimg = img_gray[hh:ha, 0:img_gray.shape[1]]
histup = cv2.calcHist([orgupimg],[0],None,[256],[0,256])
histdwn = cv2.calcHist([orgdwnimg],[0],None,[256],[0,256])
print( np.median(orgupimg), np.median(orgdwnimg) )
plt.plot(histup)
plt.show()

plt.plot(histdwn)
plt.show()

print ( getUmed( orgupimg ), getUmed( orgdwnimg ) )

"""
for slim in range( 70, 200, 10 ):
    #sharpener(img_gray, slim, "all")
    #sharpener(orgupimg, slim, "orgupimg")
    sharpener(orgdwnimg, slim, "orgdwnimgBS")
"""
