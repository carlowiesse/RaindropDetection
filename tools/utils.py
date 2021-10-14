import os
import glob
import numpy as np
import cv2

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score,precision_score,confusion_matrix


def get_paths_and_labels():

    train_folder = '../../Labelled'
    test_folder = '../../Evaluation_Data'
    labels = ['LABELLED_CLEAR_INPUTS','LABELLED_REAL_RAIN_INPUTS']

    fns = [os.path.join(train_folder,l,'input_*.png') for l in labels]
    count = [len(glob.glob(fn)) for fn in fns]

    train_img_paths = glob.glob(fns[0])+glob.glob(fns[1])
    train_img_labels = [labels[0]]*count[0]+[labels[1]]*count[1]

    fns = [os.path.join(test_folder,s+'_*.png') for s in ['left','right']]
    count = [len(glob.glob(fn)) for fn in fns]

    test_img_paths = glob.glob(fns[0])+glob.glob(fns[1])
    test_img_labels = [labels[0]]*count[0]+[labels[1]]*count[1]
    
    return train_img_paths, train_img_labels, test_img_paths, test_img_labels, labels

def array2xyz(array): return zip(*[(i,j,k) for i,row in enumerate(array) for j,k in enumerate(row)])

def normalize(array): return (array-array.min())/(array.max()-array.min())
    
def show_results(y_true,y_pred,labels):
    
    cm = confusion_matrix(y_true,y_pred,labels=labels)

    plt.figure(figsize = (5,4))
    sns.heatmap(cm, annot=True,xticklabels=['Clear','Rain'], yticklabels=['Clear','Rain'], cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    precision = precision_score(y_true,y_pred,pos_label=labels[0])
    recall = recall_score(y_true,y_pred,pos_label=labels[0])

    print('Clear Classification\n--------------------\nPrecision: %.2f - Recall: %.2f\n' % (precision, recall))

    precision = precision_score(y_true,y_pred,pos_label=labels[1])
    recall = recall_score(y_true,y_pred,pos_label=labels[1])

    print('Raindrop Classification\n--------------------\nPrecision: %.2f - Recall: %.2f\n' % (precision, recall))

def load_data(paths, size=128):
    
    images = []
    
    for p in paths:

        im = cv2.imread(p)
        im_b = im[...,0]

        gamma = 0.2
        im_gamma = im_b/255.
        im_gamma = np.power(im_gamma,gamma)
        im_gamma = np.clip(255*im_gamma,0,255)
        im_gamma = normalize(im_gamma)*255
        im_gamma = im_gamma.astype(np.uint8)
        
        im = resize_with_border(im_gamma,size)
        images.append(im)
    
    images = np.array(images)[...,np.newaxis]
    return images

def resize_with_border(im,size):
    
    old_size = im.shape[:2]
    ratio = float(size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return im
