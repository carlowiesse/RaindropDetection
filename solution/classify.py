''' PYTHON IMPORTS '''
import os
import sys
import cv2
import glob
import time
import pickle
import numpy as np
import numpy.matlib as nm
import numpy.fft as f
from sklearn.metrics import recall_score,precision_score

''' IMPORTS TO CREATE WINDOWS EXECUATBLE '''
import sklearn.utils._cython_blas
import sklearn.utils._weight_vector

'''#---------------   FEATURE EXTRACTION ALGORITHM   ----------------#'''
class GIST(): # Reference: https://github.com/imoken1122/GIST-feature-extractor

    def __init__(self,param):

        self.param = param # variable used to load feature extractor parameters

    def _prefilt(self,img): # function used to apply a pre-filter using FFTs
        
        w = 5
        fc = self.param["fc_prefilt"]
        s1 = fc/np.sqrt(np.log(2))
        img = np.log(img+1)
        img = np.pad(img,[w,w],"symmetric")

        sn,sm = img.shape
        n = np.max([sn,sm])
        n += n%2

        if sn == sm:
            img = np.pad(img,[0,int(n-sn)],"symmetric")
        else:
            img = np.pad(img,[0,int(n-sn)], "symmetric")[:,:sm]

        fx,fy = np.meshgrid(np.arange(-n/2,n/2),np.arange(-n/2,n/2))
        gf = f.fftshift((np.exp(-(fx**2+fy**2)/(s1**2))))
        gf = nm.repmat(gf,1,1)
        output = img - np.real(f.ifft2(f.fft2(img)*gf))

        localstd = nm.repmat(np.sqrt(abs(f.ifft2(f.fft2(output**2)*gf))),1,1)
        output = output/(0.2+localstd)
        output = output[w:sn-w, w:sm-w]

        return output

    def _createGabor(self,ops,n): # function used to define gabor filters using FFTs

        gabor_param = []
        Nscalse = len(ops)
        Nfilters = sum(ops)

        if len(n) == 1: n = [n[0],n[0]]

        for i in range(Nscalse):
            for j in range(ops[i]):
                gabor_param.append([.35,.3/(1.85**(i)),16*ops[i]**2/32**2, np.pi/(ops[i])*(j)])

        gabor_param = np.array(gabor_param)
        fx, fy = np.meshgrid(np.arange(-n[1]/2,n[1]/2-1 + 1), np.arange(-n[0]/2, n[0]/2-1 + 1))
        fr = f.fftshift(np.sqrt(fx**2+fy**2))
        t = f.fftshift(np.angle(fx+1j*fy))

        gabor = np.zeros([n[0],n[1],Nfilters])

        for i in range(Nfilters):
            tr = t + gabor_param[i,3]
            tr+= 2*np.pi*(tr<-np.pi) - 2*np.pi*(tr>np.pi)
            gabor[:,:,i] = np.exp(-10*gabor_param[i,0] * (fr/n[1]/gabor_param[i,1]-1)**2 - 2*gabor_param[i,2]*np.pi*tr**2)

        return gabor
        
    def _averagePooling(self,x,N): # function used to perform average pixel pooling

        nx = list(map(int,np.floor(np.linspace(0,x.shape[0],N[0]+1))))
        ny = list(map(int,np.floor(np.linspace(0,x.shape[1],N[1]+1))))
        output  = np.zeros((N[0],N[1]))

        for xx in range(N[0]):
            for yy in range(N[1]):
                a = x[nx[xx]:nx[xx+1],ny[yy]:ny[yy+1]]
                v = np.mean(np.mean(a,0))
                output[xx,yy]=v

        return output

    def _extract(self,img): # function used to compute Gist features using all of the above

        img = img.astype(np.float32)
        img = self._prefilt(img)

        w = self.param["numberBlocks"]
        G = self._createGabor(self.param["orientationsPerScale"], np.array(img.shape) + 2*self.param["boundaryExtension"])
        be = self.param["boundaryExtension"]
        ny, nx, Nfilters = G.shape
        W = w[0] * w[1]
        output = np.zeros((W*Nfilters,1))
        img = np.pad(img,[be,be],"symmetric")
        img = f.fft2(img)
        
        k = 0
        for n in range(Nfilters):
            ig = abs(f.ifft2(img*nm.repmat(G[:,:,n],1,1)))
            ig = ig[be:ny-be,be:nx-be]
            v = self._averagePooling(ig,w)
            output[k:k+W,0] = v.reshape([W,1],order="F").flatten()
            k += W

        output = np.array(output).flatten()
        return output
'''#---------------   FEATURE EXTRACTION ALGORITHM   ----------------#'''

'''#---------------   COMPLETE CLASSIFICATION PIPELINE   ----------------#'''
def pipeline(query_image):
    
    im_b = query_image[...,2] # get blue channel

    # apply gamma correction
    gamma = 0.2
    im_gamma = im_b/255.
    im_gamma = np.power(im_gamma,gamma)
    im_gamma = np.clip(255*im_gamma,0,255)
    im_gamma = (im_gamma-im_gamma.min())/(im_gamma.max()-im_gamma.min())
    im_gamma = (255*im_gamma).astype(np.uint8)
    
    # resize image
    size = 128
    old_size = im_gamma.shape[:2]
    ratio = float(size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im_gamma, (new_size[1], new_size[0]))

    # use black filling to make the image a square image
    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    # feature extraction parameters
    params = {"orientationsPerScale":np.array([8,8,8,8]),
              "numberBlocks":[4,4],
              "fc_prefilt":4,
              "boundaryExtension":4}

    # extract features
    feature_extractor = GIST(params)
    features = feature_extractor._extract(im)
    features = features.reshape(1,-1)

    # load SVM model trained with the training data
    with open('svm_model.pkl', 'rb') as fp:
        classifier = pickle.load(fp)
    
    # classify features using the SVM model
    prediction = classifier.predict(features)
    predicted_label = 'Droplets' if prediction[0] else 'Clear'
    
    return predicted_label
'''#---------------   COMPLETE CLASSIFICATION PIPELINE   ----------------#'''

'''#---------------   INPUT IMAGE LOADER   ----------------#'''
def classify(query_image_path):

    query_image = cv2.imread(query_image_path) # load image as BGR
    query_image = cv2.cvtColor(query_image,cv2.COLOR_BGR2RGB) # change format to RGB

    # measure computation time for each classification attempt
    start = time.time()
    predicted_label = pipeline(query_image)
    time_elapsed = time.time() - start

    # print results
    print('Image path: %s\nPredicted label: %s\nTime elapsed: %.2f s' % (query_image_path,predicted_label,time_elapsed))

    return predicted_label
'''#---------------   INPUT IMAGE LOADER   ----------------#'''

'''#---------------   EVALUATION SET BENCHMARK   ----------------#'''
def benchmark():

    test_folder = '../../Evaluation_Data' # relative path to the evaluation folder
    filenames = [os.path.join(test_folder,s+'_*.png') for s in ['left','right']] # load file names
    count = [len(glob.glob(fn)) for fn in filenames] # count how many there are for each label

    test_image_paths = glob.glob(filenames[0])+glob.glob(filenames[1]) # load file paths
    test_image_labels = ['Clear']*count[0]+['Droplets']*count[1] # assign labels

    # classify all files in the evaluation folder
    predicted_labels = []
    for path in test_image_paths:
        predicted_labels.append(classify(path))
        print('------------------------')

    # print recall and precision scores
    recall = recall_score(test_image_labels,predicted_labels,pos_label='Droplets')
    precision = precision_score(test_image_labels,predicted_labels,pos_label='Droplets')
    print('Droplet Classification - Recall: %.2f - Precision: %.2f' % (recall,precision))
'''#---------------   EVALUATION SET BENCHMARK   ----------------#'''

'''#---------------   USER INTERFACE   ----------------#'''
msg = 'y' # default 'msg' variable to 'y'
while msg=='y': # keep the application running as long as the user wants

    # ask for input arguments
    query_image_path = input("Enter the location of the image you want to classifiy: ")

    if query_image_path == '--benchmark': # special argument to run a benchmark on the evaluation set
        benchmark()
        break
    else: # classify the image located in the given file path
        predicted_label = classify(query_image_path)
        msg = input("Another image? (y/n)  ") # ask the user if they want to continue

# end the application
print('------------------------')
print("Thanks for using this application ^.^")
input("Press ENTER to exit")
'''#---------------   USER INTERFACE   ----------------#'''





