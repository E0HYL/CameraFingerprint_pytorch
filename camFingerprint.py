import numpy as np
#from matplotlib.image import imread,imsave
import cv2
import pywt
import matplotlib.pyplot as plt
import sys
from numba import jit,njit
from scipy.signal import correlate2d

def matshow(title,mat):
    mat = mat.astype(float)
    mi = np.min(mat)
    ma = np.max(mat)
    mat -= mi
    mat/=(ma-mi)
    mat*=255
    cv2.imshow(title,np.uint8(mat))

def rgb2gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    print('[warning] rgb2gray() not performing any process to image:',img.shape,type(img))
    return img

def Nrgb2gray(mat):
    v=np.array([.299,.587,.114])
    if len(mat.shape)==3:
        return mat.dot(v)

def DCTSplit(in_img, α):

    d_in_img = in_img.astype(float)
    d_in_img_t = cv2.dct(d_in_img)

    row = in_img.shape[0]
    cutoff = round(α * row)
    ### fliplr: flip array in left/right direction
    tr_high_t = np.fliplr(np.tril(np.fliplr(d_in_img_t), cutoff))
    tr_low_t = d_in_img_t - tr_high_t
    high = cv2.idct(tr_high_t)
    low = cv2.idct(tr_low_t)
    return low, high

@njit
def _denoiseFilter(mat,σ2_0):
    σ2 = np.zeros(mat.shape)
    '''# when no JIT
    try:
        rowN,colN = mat.shape
    except ValueError as error:
        print('[',error,'] in _denoiseFilter: shape of mat cannot be ',mat.shape,\
        '\nthis error might be raised because of an unsplited multichannel image'\
            ,sep='')
        exit()
    '''
    mat2 = mat**2
    rowN,colN = mat.shape
    '''# padding with zeros, may be faster
    padded = np.zeros((rowN+8,colN+8))
    padded[4:rowN+4,4:colN+4]=mat'''
    #estimate local variance
    for row in range(rowN):
        for col in range(colN):
            '''# when padding
            i=row+4
            j=col+4'''
            σ2_ = [0,0,0,0]
            for margin in [1,2,3,4]:
                
                rowL=row-margin
                rowR=row+margin+1
                colL=col-margin
                colR=col+margin+1
                if rowL<0:
                    rowL=0
                if rowR>rowN:
                    rowR=rowN
                if colL<0:
                    colL=0
                if colR>colN:
                    colR=colN
                N = (rowR-rowL)*(colR-colL)
                sum_ = mat2[rowL:rowR,colL:colR].sum()
                σ2_[margin-1] = sum_/N - σ2_0
                if σ2_[margin-1]<0:
                    σ2_[margin-1]=0
                '''# when padding
                rowL=i-margin
                rowR=i+margin+1
                colL=j-margin
                colR=j+margin+1
                N =(2*margin+1)**2
                sum_ = np.power(padded[rowL:rowR,colL:colR],2).sum()
                σ2_[margin-1] = sum_/N - σ2_0
                if σ2_[margin-1]<0:
                    σ2_[margin-1]=0
                 '''

            σ2[row,col]=min(σ2_)

    kernel =  σ2/(σ2+σ2_0)
    return mat*kernel

@njit
def _noiseFilter(mat,σ2_0):
    σ2 = np.zeros(mat.shape)
    '''# when no JIT
    try:
        rowN,colN = mat.shape
    except ValueError as error:
        print('[',error,'] in _denoiseFilter: shape of mat cannot be ',mat.shape,\
        '\nthis error might be raised because of an unsplited multichannel image'\
            ,sep='')
        exit()
    '''
    rowN,colN = mat.shape
    '''# padding with zeros, may be faster
    padded = np.zeros((rowN+8,colN+8))
    padded[4:rowN+4,4:colN+4]=mat'''
    mat2 = mat**2
    #estimate local variance
    for row in range(rowN):
        for col in range(colN):
            '''# when padding
            i=row+4
            j=col+4'''
            σ2_ = [0,0,0,0]
            for margin in [1,2,3,4]:
                
                rowL=row-margin
                rowR=row+margin+1
                colL=col-margin
                colR=col+margin+1
                if rowL<0:
                    rowL=0
                if rowR>rowN:
                    rowR=rowN
                if colL<0:
                    colL=0
                if colR>colN:
                    colR=colN
                N = (rowR-rowL)*(colR-colL)
                sum_ = mat2[rowL:rowR,colL:colR].sum()
                σ2_[margin-1] = sum_/N - σ2_0
                if σ2_[margin-1]<0:
                    σ2_[margin-1]=0
                '''# when padding
                rowL=i-margin
                rowR=i+margin+1
                colL=j-margin
                colR=j+margin+1
                N =(2*margin+1)**2
                sum_ = np.power(padded[rowL:rowR,colL:colR],2).sum()
                σ2_[margin-1] = sum_/N - σ2_0
                if σ2_[margin-1]<0:
                    σ2_[margin-1]=0
                 '''

            σ2[row,col]=min(σ2_)

    kernel =  σ2_0/(σ2+σ2_0)
    return mat*kernel

def wienerFilterDFT(mat,σ):
    spec = np.fft.fft2(mat)
    
    m,n = mat.shape
    specM = np.absolute(spec/np.sqrt(m*n))
    filteredM = _noiseFilter(specM,σ**2)
    zeroMask = specM==0
    specM[zeroMask]=1
    filteredM[zeroMask]=0
    spec *= filteredM/specM
    
    return np.real(np.fft.ifft2(spec))

def denoise(img,σ0=2):
    img = img.astype(float) 
    spec = pywt.wavedec2(img,'db8',level=4) # spec: [A4,(H4,V4,D4),..,(H1,V1,D1)]
    #origin=spec.copy()
    σ2_0 = σ0**2
    #spec[0] = _denoiseFilter(spec[0],σ2_0)
    filtered = [spec[0]]
    
    for tu in spec[1:]:
        subbands=[]
        for band in tu:
            subbands.append(_denoiseFilter(band,σ2_0))            
        filtered.append(tuple(subbands))
    return pywt.waverec2(filtered,'db8')

def denoise1(img):
    return cv2.medianBlur(img,5)

def getNoise(img,σ0):
    img = img.astype(float) 
    spec = pywt.wavedec2(img,'db8',level=4) # spec: [A4,(H4,V4,D4),..,(H1,V1,D1)]
    #origin=spec.copy()
    #spec[0] = _denoiseFilter(spec[0],σ2_0)
    filtered = [np.zeros(spec[0].shape)]
    
    for tu in spec[1:]:
        subbands=[]
        for band in tu:
            subbands.append(_noiseFilter(band,σ0**2))            
        filtered.append(tuple(subbands))
    
    return pywt.waverec2(filtered,'db8')

@njit
def _intenScaling(i):
    T=252
    v=6
    if i<T:
        return i/T
    else:
        return np.exp(-1*np.power(i-T,2)/v)

@njit
def _intenScale(mat,m,n):
    for i in range(m):
        for j in range(n):
            mat[i,j]=_intenScaling(mat[i,j])
    return mat

def intenScale(mat):
    ret = mat.copy().astype(float)
    m,n = mat.shape
    return _intenScale(ret,m,n)


def nonSaturationMask(mat):
    ret = np.ones(mat.shape)
    dh = mat - np.roll(mat,1,axis=1)
    dv = mat - np.roll(mat,1,axis=0)
    ret = np.logical_and(dh,dv)
    temp = np.logical_and(np.roll(dh,-1,axis=1),np.roll(dv,-1,axis=0))
    ret = np.logical_and(ret,temp)

    maxint = np.zeros(3)

    for ch in range(3):
        maxint[ch]=np.max(mat[:,:,ch])
        if maxint[ch]>250:
            ret[:,:,ch] = np.logical_or(np.logical_not(mat[:,:,ch]==maxint[ch]),ret[:,:,ch])

    return ret


def zeroMean(mat):
    (m,n,ch)=mat.shape
    row = np.zeros((m,ch))
    col = np.zeros((n,ch))
    ret = mat - np.mean(mat,axis=(0,1))
    row = np.mean(mat,axis=1)
    col = np.mean(mat,axis=0)
    ret = np.transpose(np.transpose(ret,(1,0,2))-row,(1,0,2))
    ret -= col
    return ret


def zeroMeanTotal(mat):
    ret = np.zeros(mat.shape)
    ret[0::2,0::2,:]=zeroMean(mat[0::2,0::2,:])
    ret[1::2,0::2,:]=zeroMean(mat[1::2,0::2,:])
    ret[0::2,1::2,:]=zeroMean(mat[0::2,1::2,:])
    ret[1::2,1::2,:]=zeroMean(mat[1::2,1::2,:])
    return ret

def getFingerprint(imgList,σ0=2,cropxl=0,cropxr=0,cropyt=0,cropyb=0):
    N = len(imgList)
    
    loadQ=(type(imgList[0])==str)
    if loadQ:
        m,n,ch=cv2.imread(imgList[0]).shape
    else:
        m,n,ch=imgList[0].shape
    m-=cropyt+cropyb
    n-=cropxl+cropxr
    patt = np.zeros((m,n,ch))
        
    NN=np.zeros(patt.shape)
    for i in range(N):
        if loadQ:
            imgi = cv2.imread(imgList[i])[cropyt:cropyt+m,cropxl:cropxl+n,:]
        else:
            imgi = imgList[i][cropyt:cropyt+m,cropxl:cropxl+n,:]
        
        mask = nonSaturationMask(imgi)
        
        for ch in range(3):
            noise = getNoise(imgi[:,:,ch],σ0)   
            inten = intenScale(imgi[:,:,ch])*mask[:,:,ch]
            patt[:,:,ch]+=noise*inten
            NN[:,:,ch]+=np.power(inten,2)
        print(i+1,'/',N,sep='',end='\r')
        sys.stdout.flush()
    patt = patt/(NN+1)
    #post process
    patt = zeroMeanTotal(patt)
    patt = Nrgb2gray(patt)
    sigma = np.std(patt)
    patt = wienerFilterDFT(patt,sigma)
    
    print(' '*50,end='\r')
    return patt

def noiseExtract(img,σ0=2):
    if type(img)==str:
        img = cv2.imread(img)
    
    ret = np.zeros(img.shape)
    for ch in range(3):
        ret[:,:,ch]=getNoise(img[:,:,ch],σ0)
    ret = zeroMeanTotal(ret)
    ret = Nrgb2gray(ret)
    sigma = np.std(ret)
    ret = wienerFilterDFT(ret,sigma)
    return ret

def corr(mat1,mat2):
    v1=mat1.ravel()
    v2=mat2.ravel()
    v1-=np.mean(v1)
    v2-=np.mean(v2)
    norm1=np.sqrt(v1.dot(v1))
    norm2=np.sqrt(v2.dot(v2))
    return v1.dot(v2)/norm1/norm2

def PCE_0(I,K):
    if type(I)==str:
        I = cv2.imread(I)
    W = noiseExtract(I)
    I = rgb2gray(I)
    m,n = I.shape
    #print(I.shape,W.shape,K.shape)
    return PCE(I,W,K,11)

def PCE(I,W,K,nsize=11,searchRange=[1,1]):
    m,n = I.shape
    X = I*K
    X -= np.mean(X)
    Y = W - np.mean(W)
    
    #XY = correlate2d(X,Y,mode = 'same')
    XY = _Ncorr2(X,Y,m,n)
    XYinRange = XY[-searchRange[0]:,-searchRange[1]:]
    XYpeak = np.max(XYinRange) #np.dot(X.ravel(),Y.ravel())**2
    (peaki,peakj) = np.unravel_index(np.argmax(XYinRange),XYinRange.shape)
    peaky = searchRange[0]-peaki-1
    peakx = searchRange[1]-peakj-1
    peaki = m-1 - peaky
    peakj = n-1 - peakx
    #print('peakheight:',format(XYpeak,'.2E'))
    sgn = np.sign(XYpeak)
    margin = nsize//2
    neighbor = np.sum(np.roll(XY,(-peaki+margin,-peakj+margin),(0,1))[0:nsize,0:nsize]**2)
    #print('neighbor:',neighbor)    
    PCE_energy = (np.sum(XY**2) - neighbor)/(m*n-nsize**2)
    return (sgn*XYpeak**2/PCE_energy,XYpeak,peakx,peaky)

#for fingerprint vs fingerprint
def PCE_fp(X,Y,nsize=11,searchRange=[1,1]):
    m,n = X.shape
    X -= np.mean(X)
    Y -= np.mean(Y)  
    #XY = correlate2d(X,Y,mode = 'same')
    XY = _Ncorr2(X,Y,m,n)
    XYinRange = XY[-searchRange[0]:,-searchRange[1]:]
    XYpeak = np.max(XYinRange) #np.dot(X.ravel(),Y.ravel())**2
    (peaki,peakj) = np.unravel_index(np.argmax(XYinRange),XYinRange.shape)
    peaky = searchRange[0]-peaki-1
    peakx = searchRange[1]-peakj-1
    peaki = m-1 - peaky
    peakj = n-1 - peakx
    #print('peakheight:',format(XYpeak,'.2E'))
    sgn = np.sign(XYpeak)
    margin = nsize//2
    neighbor = np.sum(np.roll(XY,(-peaki+margin,-peakj+margin),(0,1))[0:nsize,0:nsize]**2)
    #print('neighbor:',neighbor)    
    PCE_energy = (np.sum(XY**2) - neighbor)/(m*n-nsize**2)
    return (sgn*XYpeak**2/PCE_energy,XYpeak,peakx,peaky)

@njit
def _corr2(mat1,mat2,m,n):
    ret=np.zeros((m,n))
    for i in range(m):
        rowl1 = i - m//2
        rowr1 = rowl1+m
        rowl2 = -rowl1
        rowr2 = rowl2+m
        if rowl1<0:
            rowl1=0
        if rowr1>m:
            rowr1=m
        if rowl2<0:
            rowl2=0
        if rowr2>m:
            rowr2=m
        for j in range(n):
            coll1=j-n//2
            colr1=coll1+n
            coll2=-coll1
            colr2=coll2+n
            if coll1<0:
                coll1=0
            if colr1>n:
                colr1=n
            if coll2<0:
                coll2=0
            if colr2>n:
                colr2=n
            ret[i,j]=np.sum(mat1[rowl1:rowr1,coll1:colr1]*mat2[rowl2:rowr2,coll2:colr2])
    return ret

def _Ncorr2(mat1,mat2,m,n):
    mat1 -= np.mean(mat1)
    mat2 -= np.mean(mat2)
    mat2 = mat2[::-1,::-1]
    f1 = np.fft.fft2(mat1)
    f2 = np.fft.fft2(mat2)
    return np.real(np.fft.ifft2(f1*f2))