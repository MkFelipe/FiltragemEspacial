import numpy as np
import Image
from scipy import misc
import matplotlib.pyplot as plt
from noise_generator import *
#from test_noise_generator import *
from scipy import misc
import math as math
import time



def kernels(type,dim,im):
    kernel = []
    im = np.array(im)
    if ((type == 'gaussian')&(dim <= 7)):
        '''Make a square gaussian kernel.
        dim is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.'''
           
        a = raw_input("type a number for fwhm")
        fwhm =  float(a)

        #fwhm = 3
        center = None
 
        x = np.arange(0, dim, 1, float)
        y = x[:,np.newaxis]
    
        if center is None:
            x0 = y0 = dim // 2
        else:
            x0 = center[0]
            y0 = center[1]
                                          
        result = np.exp(-2*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2) #that is the gaussian function
        kernel = result
        kernel = np.array(kernel)
        kernel = kernel.reshape(dim,dim)
        return kernel

    if ((type == 'mean')&(dim <= 7)):
        '''Create a mean kernel with [dim,dim]'''
        kernel = np.ones((dim,dim),np.float32)/(dim*dim)
        return kernel

            
    
    if ((type == 'median')&(dim <= 21)):
        x = im.shape[0]
        y = im.shape[1]
        size = dim//2
        k = []
        kc = []
        kd = []
        kb = []
        ke = []
        kernel = [im,im,im,im,im]
        l = im.shape[0]+(2*size) 
        c = im.shape[1]+(2*size) 
        '''create the images to the convolutions to save on "kernel"'''
        for i in range(l*c):
            k.append(0)
            kc.append(0)
            kd.append(0)
            kb.append(0)
            ke.append(0)
        '''reshape the images'''    
        k = np.array(k)
        k = k.reshape(l,c)
        kc = np.array(k)
        kc = kc.reshape(l,c)
        kd = np.array(k)
        kd = kd.reshape(l,c)
        kb = np.array(k)
        kb = kb.reshape(l,c)
        ke = np.array(k)
        ke = ke.reshape(l,c)
        '''Create a "kernel", a vector of convolved images'''
        for i in range (size, x-size):
            for j in range (size, y-size):
                k[i][j] = im[i][j]
        for i in range (0, x-size):
            for j in range (size, y-size):
                kc[i-size][j] = im[i][j]
        for i in range (size, x-size):
            for j in range (size, y-size):
                kd[i][j+size] = im[i][j]
        for i in range (size, x-size):
            for j in range (size, y-size):
                kb[i+size][j] = im[i][j]
        for i in range (size, x-size):
            for j in range (0, y-size):
                ke[i][j-size] = im[i][j]
                        
        kernel[0] = k
        kernel[1] = kc
        kernel[2] = kd
        kernel[3] = kb
        kernel[4] = ke
        return kernel
    

def resizeim(im,dim):
    '''put a black border in the image, to make the convolutions'''
    size = dim//2
    new_im = []
    x,y = im.shape
    
    l = im.shape[0]+(2*size) 
    c = im.shape[1]+(2*size) 
    
    for i in range(l*c):
            new_im.append(0)
    new_im = np.array(new_im)
    new_im = new_im.reshape(l,c)
    for i in range(0,x):
    	for j in range(0,y):
    	    new_im[i+size][j+size] = im[i][j] 
    return new_im



def convolutions (type, im, k, dim):
    '''Make the convolutions'''
    x = im.shape[0]
    y = im.shape[1]

    size = dim//2
    #print size
    #print kernel
    aux1 = []
    aux2 = []


    if ((type == 'gaussian')&(dim <= 7)):
        '''apply the gaussian kernel on the images'''
        new_im = im
        for i in range (size, x-size):
                for j in range (size, y-size):
                    aux1 = im[i-size:i+size+1,j-size:j+size+1]#get the slices of the image''
                    aux2 = aux1 * kernel
                    aux2 = aux2.sum()
                    aux3 = int(aux2)
                    new_im[i][j] = aux3
        return new_im
          
    if ((type == 'mean')&(dim <= 7)):
        '''apply the mean kernel on the images'''
        new_im = im
        for i in range (size, x-size):
            for j in range (size, y-size):
                aux1 = im[i-size:i+size+1,j-size:j+size+1] #get the slices of the image'''
                aux2 = aux1 * kernel
                aux2 = aux2.sum()
                new_im[i][j] = aux2        

        return new_im
            
    
    if ((type == 'median')&(dim <= 21)):
        '''get the image array and apply the np.median on it'''
        new_im = np.median(kernel, 0)
        print new_im
            
        return new_im         
    

#-------------------main-------------------------------
lena = misc.lena() #import lena from misc
im_gauss = add_noise(lena,'normal', sigma=20) #add a gaussian noise
im_sp = add_noise(lena,'salt-and-pepper',pa=0.05, pb=0.05) #add a salt & pepper noise
g="gaussianruid.jpg"
sp="saltandpepperruid.jpg"
misc.imsave(g,im_gauss) #save the noise image 
misc.imsave(sp, im_sp)  #save the noise image

tipo = raw_input('write a type for the filter (median, gaussian or mean)')
s = raw_input ('write a number for the kernel dimensions')
dim = float(s)


'''where the magics happen'''
if (tipo == 'mean'):
    # gaussian noise
    begin = time.time()
    kernel = kernels(tipo, dim, lena)
    im_gauss = resizeim(im_gauss, dim)
    im_gauss = convolutions(tipo, im_gauss, kernel, dim)
    end = time.time()
    print "gaussian time: ", end-begin
    # salt & pepper noise
    begin = time.time()
    kernel = kernels(tipo, dim, lena)
    im_sp = resizeim(im_sp, dim)
    im_sp = convolutions(tipo, im_sp, kernel, dim)
    end = time.time()
    print "salt & pepper time: ", end-begin
    
if (tipo == 'median'):
    # gaussian noise
    begin = time.time()
    kernel = kernels(tipo, dim, im_gauss)
    im_gauss = convolutions(tipo, im_gauss, kernel, dim)
    end = time.time()
    print "gaussian time: ", end-begin
    # salt & pepper noise
    begin = time.time()
    kernel = kernels(tipo, dim, im_sp)
    im_sp = convolutions(tipo, im_sp, kernel, dim)
    end = time.time()
    print "salt & pepper time: ", end-begin


if (tipo == 'gaussian'):
    # gaussian noise
    kernel = kernels(tipo, dim, lena)
    begin = time.time()
    im_gauss = resizeim(im_gauss, dim)
    im_gauss = convolutions(tipo, im_gauss, kernel, dim)
    end = time.time()
    print "gaussian time: ", end-begin
    # salt & pepper noise
    begin = time.time()
    im_sp = resizeim(im_sp, dim)
    im_sp = convolutions(tipo, im_sp, kernel, dim)
    end = time.time()
    print "salt & pepper time: ", end-begin



gauss = "image_gaussian.jpeg"
salt = "image_saltandpepper.jpeg"

misc.imsave(gauss,im_gauss) #Save the filtered image

misc.imsave(salt, im_sp)  #Save the filtered image