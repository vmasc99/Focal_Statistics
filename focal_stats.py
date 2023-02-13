# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 17:00:24 2022

@author: valen
"""


### import packages

from PIL import Image
import numpy as np
import warnings
from tqdm import tqdm, trange
import time
import matplotlib.pyplot as plt


def FocalStatistics():
    
    height = None
    width = None
    radius = None
    angle = None
    angle_from = None
    angle_to = None
    i_radius = None
    o_radius = None
    
    
    stats = "\nThe available statistics are: 'mean', 'max', 'min', 'sd', 'var'(variance), 'majority', 'variety'.\n\n"
    
    shapes = "\nThe available shapes are: 'square', 'circle', 'rectangle', 'wedge' or 'annulus'.\n\n"
    
    all_ = shapes + "\n" + stats +"\n\n"
    
    welcome = input("\nIf you want to know which are the available shapes, input ?shapes, for statistics, ?stats, for both, ?all. \nAny other input will make it skip.\n\n")
    
    if welcome == "?stats":
        
        print(stats)
        
    elif welcome == "?shapes":
        
        print(shapes)
        
    elif welcome == "?all":
        
        print(all_)
        
    else:
        pass
    
    shape = str(input("\nPlease input a shape between square, circle, annulus, wedge or rectangle: \n\n")).lower()
    
    if shape == "circle":
        
        radius = int(input(f"Please enter the radius of the {shape} (integer only): "))
        
    elif shape == "annulus":
            
        i_radius = int(input(f"Please enter the inner radius of the {shape} (integer only): "))
        
        o_radius = int(input(f"Please enter the outer radius of the {shape} (integer only): "))
        
    elif shape == "rectangle":
        
        width = int(input("\nPlease enter the width of the rectangle (integer only): "))
        
        height = int(input("\nPlease enter the height of the rectangle (integer only): "))
        
    elif shape == "square":
        
        width = int(input("\nPlease enter the side of the square (integer only): "))
    
      
    elif shape == "wedge":
        
        radius = int(input(f"Plese enter the radius of the {shape} (integer only): "))
        
        angle_from = int(input("\nPlease enter the starting angle of the wedge: "))
        
        angle_to = int(input("\nPlease enter the ending angle of the wedge: "))
    else:
        raise ValueError("Shape is not available. Choose between 'square', 'circle', 'rectangle', 'wedge' or 'annulus'.")
        
        
     ## functions calculating statistics of masked values
    
    def fun_mean(pxls_masked):
        return np.mean(pxls_masked)
    
    def fun_max(pxls_masked):
        return np.max(pxls_masked)
    
    def fun_min(pxls_masked):
        return np.min(pxls_masked)
    
    def fun_sd(pxls_masked):
        return np.std(pxls_masked)
    
    def fun_var(pxls_masked):
        return np.var(pxls_masked)
    
    def fun_majority(pxls_masked):
        return np.bincount(pxls_masked).argmax()
    
    def fun_variety(pxls_masked):
        return len(np.unique(pxls_masked.flatten()))
    
    method = str(input("\nPlease input the statistics to be performed: \n")).lower()
    
    if method == 'mean':
            fun_calc = fun_mean
    elif method == 'max':
            fun_calc = fun_max
    elif method == 'min':
            fun_calc = fun_min
    elif method == 'sd':
            fun_calc = fun_sd
    elif method == 'var':
            fun_calc = fun_var
    elif method == 'majority':
            fun_calc = fun_majority
    elif method == 'variety':
            fun_calc = fun_variety
    else:
        raise ValueError("Method is not available. Choose between 'mean', 'max', 'min', 'sd', 'var', 'majority', 'variety'.")

    # load image
    
    image_import = False
    
    while image_import == False:
        
        
        path = str(input("\nPlease enter the path where the image chosen is at and its name: \n"))
        
        try:
            if path[-3:] == 'tif':
                img = Image.open(path)
              
                img_np = np.asarray(img)
                
                img_np = img_np[:,:,0:3]
                    
                image_import = True
                
                
                    
                  
               
            elif path[-3:] == 'jpg' or 'jpeg' or 'png':
                
                img = Image.open(path)
              
                img_np = np.asarray(img)

                image_import = True
                
                
            
        except:
            print(f"\nNo such file or directory: {path} \n")
    
    print("\nImage successfully imported.\n\n")
    
    plt.imshow(img)
    plt.show()
    
    #img_np = np.asarray(img)
    
    
    
    print("\nStarting Focal Statistics.\n\n")
    
    ### functions
    
    ## functions returning values of mask shape
    
    def mask_square(w: int, idx: list, img_arr):
        
        h = w
        
        if(h % 2 == 0 or w % 2 == 0):
            warnings.warn("Input is an even number, square can't be centered correctly and is shifted one pixel to the left.")
        
        arr_shape = img_arr.shape
        
        y1 = int(idx[0] - np.floor(h/2))
        y2 = int(idx[0] + np.ceil(h/2))
        
        x1 = int(idx[1] - np.floor(w/2))
        x2 = int(idx[1] + np.ceil(w/2))
        
        if y1 < 0:
            y1 = 0
        if y2 > arr_shape[0]:
            y2 = arr_shape[0]
        if x1 < 0:
            x1 = 0
        if x2 > arr_shape[1]:
            x2 = arr_shape[1]
        
        pxls_masked = img_arr[y1:y2, x1:x2, :]
        pxls_masked_shape = pxls_masked.shape
        
        return pxls_masked.reshape(pxls_masked_shape[0] * pxls_masked_shape[1], arr_shape[2])
    
    def mask_rectangle(h: int, w: int, idx: list, img_arr):
        
        if(h % 2 == 0 or w % 2 == 0):
            warnings.warn("Input is an even number, rectangle can't be centered correctly and is shifted one pixel.")
        
        arr_shape = img_arr.shape
        
        y1 = int(idx[0] - np.floor(h/2))
        y2 = int(idx[0] + np.ceil(h/2))
        
        x1 = int(idx[1] - np.floor(w/2))
        x2 = int(idx[1] + np.ceil(w/2))
        
        if y1 < 0:
            y1 = 0
        if y2 > arr_shape[0]:
            y2 = arr_shape[0]
        if x1 < 0:
            x1 = 0
        if x2 > arr_shape[1]:
            x2 = arr_shape[1]
        
        pxls_masked = img_arr[y1:y2, x1:x2, :]
        pxls_masked_shape = pxls_masked.shape
        
        return pxls_masked.reshape(pxls_masked_shape[0] * pxls_masked_shape[1], arr_shape[2])
    
    def mask_circle_init(radius: int, img_arr):
        
        x_center = img_arr.shape[1] / 2
        y_center = img_arr.shape[0] / 2
        
        h, w = img_arr.shape[:2]
    
        Y, X = np.ogrid[:h, :w]
        dist_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    
        mask = dist_center <= radius
    
        mask_idx = np.nonzero(mask == True)
        
        mask_idx_x_init = mask_idx[1] - img_arr.shape[1] / 2
        mask_idx_y_init = mask_idx[0] - img_arr.shape[0] / 2
        
        mask_idx_init = (mask_idx_y_init, mask_idx_x_init)
    
        return mask_idx_init
    
    def mask_wedge_init(radius: int, angle_from: int, angle_to: int):
        
        sa = angle_from
        ea = angle_to
        
        mask = np.array([0]*(2*radius+1)*(2*radius+1)).reshape(2*radius+1, 2*radius+1)
    
        c = int(np.ceil(mask.shape[0]/2))
    
        mask[c-1,c-1] = 1
    
        for r in range(radius):
    
            for t in range(sa, ea):
    
                k = int(np.floor(r * np.cos(np.radians(abs(- t - 270))) + c))
                l = int(np.floor(r * np.sin(np.radians(abs(- t - 270))) + c))
                
                mask[k, l] = 1
    
        mask_idx = np.nonzero(mask == 1)
        
        mask_idx_x_init = mask_idx[1] - np.ceil(mask.shape[0]/2)
        mask_idx_y_init = mask_idx[0] - np.ceil(mask.shape[0]/2)
        
        mask_idx_init = (mask_idx_y_init, mask_idx_x_init)
        
        return mask_idx_init
    
    def mask_shape_shift(mask_idx: tuple, idx: list, img_arr):
        
        mask_y_new = mask_idx[0] + idx[0]
        mask_x_new = mask_idx[1] + idx[1]
        
        y_outside = np.array(mask_y_new > img_arr.shape[0] - 1) | np.array(mask_y_new < 0)
        x_outside = np.array(mask_x_new > img_arr.shape[1] - 1) | np.array(mask_x_new < 0)
        
        idx_outside = np.array((y_outside) | (x_outside))
        
        mask_y_crop = mask_y_new[np.invert(idx_outside)]
        mask_x_crop = mask_x_new[np.invert(idx_outside)]
    
        return img_arr[mask_y_crop.astype(int), mask_x_crop.astype(int), :]
    
    

    
    def mask_annulus_init(i_radius: int, o_radius: int, img_arr):
        
        x_center = img_arr.shape[1] / 2
        y_center = img_arr.shape[0] / 2
        
        h, w = img_arr.shape[:2]
    
        Y, X = np.ogrid[:h, :w]
        dist_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    
        mask_o =  np.array([dist_center <= o_radius])
        
        mask_i =  np.array([dist_center >= i_radius])
        
        mask = mask_i * mask_o
        
        mask = mask.reshape(img_arr.shape[0], img_arr.shape[1])
        
        #mask = dist_center <= o_radius
        
        #mask = mask_o- mask_i
    
        mask_idx = np.nonzero(mask == True)
        
        mask_idx_x_init = mask_idx[1] - img_arr.shape[1] / 2
        mask_idx_y_init = mask_idx[0] - img_arr.shape[0] / 2
        
        mask_idx_init = (mask_idx_y_init, mask_idx_x_init)
    
        return mask_idx_init
    
    
    
    
   
    
    
    
    ### function for iterating over all pixels
    
    def image_filter(img, shape, method, h = None, w = None, radius = None, angle = None, angle_from = None, angle_to = None, i_radius = None, o_radius = None):
        
        if shape == 'square':
            fun_shape = mask_square
        elif shape == 'circle':
            fun_shape = mask_shape_shift
        elif shape == 'rectangle':
            fun_shape = mask_rectangle
        elif shape == 'wedge':
            fun_shape = mask_shape_shift
        elif shape == 'annulus':
            fun_shape = mask_shape_shift
        else:
            raise ValueError("Shape is not available. Choose between 'square', 'circle', 'rectangle', 'wedge' or 'annulus'.")
        

        
        img_shape = img.shape
        
        # idx_combs = np.array(np.meshgrid(np.arange(0,img_shape[0]), np.arange(0,img_shape[1]))).T.reshape(-1,2)
        
        img_filtered = np.zeros((img_shape[0], img_shape[1], img_shape[2]))
        
        if shape == 'circle':
            mask_idx_init = mask_circle_init(radius = radius, img_arr = img)
        elif shape == 'wedge':
            mask_idx_init = mask_wedge_init(radius = radius, angle_from = angle_from, angle_to = angle_to)
        elif shape == 'annulus':
            mask_idx_init = mask_annulus_init(i_radius = i_radius, o_radius = o_radius, img_arr = img)
        
        for i in trange(img_shape[0]):
            for j in range(img_shape[1]):
                
                if shape == 'square':
                    values_masked = fun_shape(w = w, idx = [i,j], img_arr = img)
                elif shape == 'rectangle':
                    values_masked = fun_shape(h = h, w = w, idx = [i,j], img_arr = img)
                elif shape == 'circle':
                    values_masked = fun_shape(mask_idx = mask_idx_init, idx = [i,j], img_arr = img)
                elif shape == 'wedge':
                    values_masked = fun_shape(mask_idx = mask_idx_init, idx = [i,j], img_arr = img)
                elif shape == 'annulus':
                    values_masked = fun_shape(mask_idx = mask_idx_init, idx = [i,j], img_arr = img)
                
                img_filtered[i, j, 0] = fun_calc(values_masked[:,0])
                img_filtered[i, j, 1] = fun_calc(values_masked[:,1])
                img_filtered[i, j, 2] = fun_calc(values_masked[:,2])
        
        return img_filtered
    
    
    
    img_filtered = image_filter(img = img_np, shape = shape, method = method, h = height, w = width, radius = radius, angle = angle, angle_from = angle_from, angle_to = angle_to, i_radius = i_radius, o_radius = o_radius ).astype(np.uint8)
    
    img_pil_filtered = Image.fromarray(img_filtered)

    plt.imshow(img_pil_filtered)
    
    plt.show()

    resp = input("\n\nDo you want to save the image? Type yes to do so, type anything else to end.\n")

    if resp == 'yes':
        
        loop = True
      
        while loop == True:
            
            img_name = str(input("\nPlease insert the output image path, name and its format (ex. images/image.jpg): \n"))

            try:
                img_pil_filtered.save(img_name)
                print(f"The image has been saved as {img_name}")
                loop = False
            except:
                print("\nThe path is not valid, insert another path.\n")
        
                
            
    else:
        pass
    

    
    

FocalStatistics()

    
  
    

