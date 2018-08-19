from timeit import default_timer as timer
import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv(image, filt,clip):
    res=np.empty(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
           try:
                sl=image[i:i+filt.shape[0],j:j+filt.shape[1],0]                
                sl=sl*filt                
                res[i,j,0] = sl.sum()                       
           except:
                pass            
    res[:,:,2]=res[:,:,1]=res[:,:,0] 
    if not clip:
        return res
    else:
        return res.clip(0,255).astype('uint8')

img = cv2.imread('C:/Users/Ankesh N. Bhoi/Desktop/CVIP/lena_gray.jpg')
# =============================================================================
# cv2.imshow('image',img)
# cv2.waitKey(0)
# =============================================================================
sobelx = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
sobely = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")
sharpen=np.array((
	[0, -1, 0],
	[-1, 4, -1],
	[0, -1, 0]), dtype="int")


#opx  = cv2.filter2D(img,-1,sobelx)
#opy  = cv2.filter2D(img,-1,sobely )
opx  = conv(img,sobelx,True)
opy  = conv(img,sobely,True)
non=[opx,opy]

#cv2.namedWindow("imagex")
cv2.imshow('imagex',opx)
#cv2.waitKey(0)
#cv2.namedWindow("imagey")
cv2.imshow('imagey',opy)
#cv2.waitKey(0)

op=np.empty((opx.shape[0],opx.shape[1],3),'int')
opt=np.empty((opx.shape[0],opx.shape[1],3),'int')
for i in range(opx.shape[0]):
    for j in range(opx.shape[1]):
       op[i,j,1]=op[i,j,2]=op[i,j,0]=(opx[i,j,0]**2+opy[i,j,0]**2)**.5

opt[:,:,0]=(opx[:,:,0]**2+opy[:,:,0]**2)**.5
opt[:,:,1]=opt[:,:,2]=opt[:,:,0]
op=np.array(op,'uint8')
opt=np.array(opt,'uint8')
not_sep=op


cv2.imshow('edges',op)

#cv2.waitKey(0)

# =============================================================================
# Linearly seperable filtering
# =============================================================================
sobely_col = np.array((
	[1],
	[2],
	[1]), dtype="int")

sobely_row = np.array((
        [-1, 0, 1],), dtype="int")

opy=conv(img,sobely_row,False)
opy=conv(opy,sobely_col,True)
cv2.imshow('sobely',opy)
#cv2.waitKey(0)

sobelx_col = np.array((
	[-1],
	[0],
	[1]), dtype="int")

sobelx_row = np.array((
        [1, 2, 1],), dtype="int")
opx=conv(img,sobelx_col,False )
opx=conv(opx,sobelx_row,True)
cv2.imshow('sobelx',opx)

op=np.empty((opx.shape[0],opx.shape[1],3),'int')
for i in range(opx.shape[0]):
    for j in range(opx.shape[1]):
       op[i,j,1]=op[i,j,2]=op[i,j,0]=(opx[i,j,0]**2+opy[i,j,0]**2)**.5
op=np.array(op,'uint8')
cv2.imshow('sobel',op)

cv2.imshow('diff',not_sep-op)

sep=op
cv2.waitKey(0)

# =============================================================================
# Timer
# =============================================================================
timec=[]
timesc=[]
#i=2
#f=np.ones((10**i,10**i),'int')
#r=np.ones((10**i,1),'int')
#c=np.ones((1,10**i),'int')   
r=np.random.randint(0,255,(101,1))
c=np.random.randint(0,255,(1,101))
f=r*c
start=timer()
conv(img,f,True)
end=timer()
timec.append(end-start)

start=timer()
conv(img,r,False)
conv(img,c,True)
end=timer()
timesc.append(end-start)

timec=np.reshape(timec,(1,len(timec)))
timesc=np.reshape(timesc,(1,len(timesc)))
ratio=timec/timesc
# =============================================================================
# Histogram Equalization greyscale
# =============================================================================
img = cv2.imread('C:/Users/Ankesh N. Bhoi/Desktop/CVIP/Unequalized_Hawkes_Bay_NZ.jpg')
#img = cv2.imread('C:/Users/Ankesh N. Bhoi/Desktop/CVIP/scan.png')

cv2.imshow('orignal',img)
histogram={}
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        try:
            histogram[img[i,j,0]]+=1
        except:
            histogram[img[i,j,0]]=1

intensities=sorted(histogram.keys())
frequencies=[histogram[i] for i in intensities]

plt.bar(intensities,frequencies)
plt.xlabel('Intensities')
plt.ylabel('Frequencies')
plt.title('Histogram')
plt.show()

cumulative_freq=[]
s=0
for i in frequencies:
    s+=i
    cumulative_freq.append(s)

plt.bar(intensities,cumulative_freq)
plt.xlabel('Intensities')
plt.ylabel('Frequencies')
plt.title('Cumulative Histogram')
plt.show()
    
cumulative_histogram={}
for i in range(len(cumulative_freq)):
        cumulative_histogram[intensities[i]]=cumulative_freq[i]
        
        
norm=255/((img.shape[0]*img.shape[1])-cumulative_freq[0])
for i in cumulative_histogram.keys():
    cumulative_histogram[i]=int((cumulative_histogram[i]-cumulative_freq[0])*norm)

transformed=[cumulative_histogram[i] for i in intensities]

plt.plot(intensities,transformed)
plt.xlabel('orignal intensities')
plt.ylabel('Transformed intensities')
plt.title('Transformation Function')
plt.show()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j,1]=img[i,j,2]=img[i,j,0]=cumulative_histogram[img[i,j,0]]
        
plt.bar(sorted(cumulative_histogram.values()),frequencies)
plt.xlabel('Intensities')
plt.ylabel('Frequencies')
plt.title('Histogram of transformed image')
plt.show()
        
cv2.imshow('histogram eq',img)
cv2.waitKey(0)


