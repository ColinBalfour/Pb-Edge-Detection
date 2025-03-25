#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.cluster.vq import kmeans2

def main(im_idx=3):

    color_im = cv2.imread(f'BSDS500/Images/{im_idx}.jpg')
    # color to grey
    im = cv2.cvtColor(color_im, cv2.COLOR_BGR2GRAY)
    # plt.imshow(im)
    # plt.show()

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    
    def create_gaussian(sigma, sigma_y=None):
        if sigma_y is None:
            sigma_y = sigma
            
        size = int(6 * max(sigma, sigma_y)) + 1

        gaussian = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma_y).T
        
        return gaussian
    
    def create_dog(sigma, theta, transpose=False, sigma_y=None):
        sobel = np.array([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ])
        sobel = sobel.T if transpose else sobel
        gaussian = create_gaussian(sigma, sigma_y)
        convolution = signal.convolve2d(gaussian, sobel, 'same')
        return ndimage.rotate(convolution, theta, reshape=False)
    
    # out = create_dog(2, 0)
    # plt.imshow(out, cmap='gray')
    # plt.show()
    
    
    def create_filter_bank(axes, filter, o, s, start_row=0, start_col=0, flipped=False):
        filters = []
        # Loop through sigma and theta values, create the filters, and tile them
        for i, sigma in enumerate(s):
            for j, theta in enumerate(o):
                out = filter(sigma, theta)
                filters.append(out)  # Collect all filters into a list
                
                if flipped:
                    ax = axes[start_row + j, start_col + i]
                else:
                    ax = axes[start_row + i, start_col + j]
                    
                ax.imshow(out, cmap='gray')
                ax.axis('off')  # Turn off axes for better aesthetics

        return filters  # Return the list of filters
                
    # Create a figure with a grid for all images
    size = (2, 16)
    fig, axes = plt.subplots(*size, figsize=(32, 4))
    
    o = [i * 360 / size[1] for i in range(size[1])]
    s = [(i)**2+1 for i in range(size[0])]
        
    DoG = create_filter_bank(axes, create_dog, o, s)
    
    # Adjust spacing to prevent overlap
    plt.tight_layout()

    # Save the result as an image
    plt.savefig('outputs/filter/DoG.png', dpi=300)
    plt.close()  # Close the figure after saving
    

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """

    def create_ddog(sigma, theta, transpose=False, sigma_y=None):
        sobel = np.array([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ])
        sobel = sobel.T if transpose else sobel
        
        gaussian = create_gaussian(sigma, sigma_y)
        convolution = signal.convolve2d(gaussian, sobel, 'same')
        convolution = signal.convolve2d(convolution, sobel, 'same')
        return ndimage.rotate(convolution, theta, reshape=False)
    
    def create_log(sigma, theta):
        size = int(6 * sigma) + 1
        sobel = np.array([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ])
        
        gaussian = create_gaussian(sigma)
        convolution = signal.convolve2d(gaussian, sobel, 'same')
        convolution_x = signal.convolve2d(convolution, sobel, 'same')
        
        convolution = signal.convolve2d(gaussian, sobel.T, 'same')
        convolution_y = signal.convolve2d(convolution, sobel.T, 'same')
        
        return ndimage.rotate(convolution_x + convolution_y, theta, reshape=False)
    
    # out = create_ddog(5, 0)
    # plt.imshow(out, cmap='gray')
    # plt.show()


    def create_LM_bank(large=False):

        fig, axes = plt.subplots(4, 12, figsize=(24, 8))
        
        size = (3, 6)
        
        o = [i * 180 / size[1] for i in range(size[1])]
        s = np.array([np.sqrt(2) ** i for i in range(size[0])])
        
        if large:
            s *= np.sqrt(2)

        # Call create_filter_bank for the first and second filter bank
        dog = create_filter_bank(axes, lambda x, y: create_dog(x, y, True, x * 3), o, s, start_col=0)  # First filter bank in the first half
        ddog = create_filter_bank(axes, lambda x, y: create_ddog(x, y, True, x * 3), o, s, start_col=6)  # Second filter bank in the second half
        
        size = (4, 1)
        
        o = [i * 180 / size[1] for i in range(size[1])]
        s = np.array([np.sqrt(2) ** i for i in range(size[0])])
        
        if large:
            s *= np.sqrt(2)
        
        LoG = create_filter_bank(axes, create_log, o, s, start_row=3, flipped=True)  # Second filter bank in the second half
        LoG_3 = create_filter_bank(axes, create_log, o, s * 3, start_row=3, start_col=4, flipped=True)  # Second filter bank in the second half
        gauss = create_filter_bank(axes, create_gaussian, o, s, start_row=3, start_col=8, flipped=True)  # Second filter bank in the second half
        

        # Adjust spacing to prevent overlap
        plt.tight_layout()

        # Save the result as an image
        name = 'outputs/filter/LML.png' if large else 'outputs/filter/LMS.png'
        plt.savefig(name, dpi=300)
        
        plt.close()  # Close the figure after saving
        
        return dog + ddog + LoG + LoG_3 + gauss
    
    LMS = create_LM_bank(False)
    LML = create_LM_bank(True)
    

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    # gabor = cv2.getGaborKernel((21, 21), 5, 0, 10, 1)
    # plt.imshow(gabor, cmap='gray')
    # plt.show()
    
    def create_gabor(sigma, theta, lambda_, gamma, psi):
        sigma_x = sigma
        sigma_y = sigma / gamma
        size = int(6 * max(sigma_x, sigma_y)) + 1
        
        gaussian = cv2.getGaussianKernel(size, sigma_x) @ cv2.getGaussianKernel(size, sigma_y).T
        sinusoid_real = np.sin(2 * np.pi * np.arange(-(size//2), (size//2)+1) / lambda_ + psi)
        gabor = gaussian * (sinusoid_real)
        
        return ndimage.rotate(gabor, theta, reshape=False)
    
    lambda_ = 10
    gamma = 1
    psi = 0
    
    fig, axes = plt.subplots(5, 8, figsize=(16, 10))
    
    size = (5, 8)
    o = [i * 360 / size[1] for i in range(size[1])]
    s = [2 * (i+1) for i in range(size[0])]
    
    gabor = create_filter_bank(axes, lambda x, y: create_gabor(x, y, lambda_, gamma, psi), o, s)
    
    # Adjust spacing to prevent overlap
    plt.tight_layout()
    
    # Save the result as an image
    plt.savefig('outputs/filter/Gabor.png', dpi=300)
    
    plt.close()  # Close the figure after saving
    


    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    
    def create_half_disc(radius, theta):
        size = 2 * radius + 1
        half_disc = np.zeros((size, size))
        
        # Calculate the center of the disk
        cx, cy = radius, radius
        
        for i in range(size):
            for j in range(size):
                # Coordinates relative to the center of the disk
                x = i - cx
                y = j - cy
                
                # Check if the point is inside the circle
                if x**2 + y**2 <= radius**2:
                    
                    # Check if the point lies in the correct half-disk based on the angle theta
                    angle = np.arctan2(y, x)
                    if angle >= 0 and angle <= np.pi:
                        half_disc[i, j] = 1
                    else:
                        half_disc[i, j] = 0
                    
        rot = ndimage.rotate(half_disc, 180 + theta, reshape=False)
        thres = .3
        rot[rot > thres] = 1
        rot[rot <= thres] = 0
        return rot
    
    fig, axes = plt.subplots(3, 16, figsize=(32, 6))
    
    disc_size = (3, 16)
    r = [2 ** (i+1) for i in range(disc_size[0])]
    theta = [i * 360 / disc_size[1] for i in range(disc_size[1])]
    
    half_disc = create_filter_bank(axes, create_half_disc, theta, r)
    
    # Adjust spacing to prevent overlap
    plt.tight_layout()
    
    # Save the result as an image
    plt.savefig('outputs/filter/HDMasks.png', dpi=300)
    
    plt.close()  # Close the figure after saving
    
    print("Done with half-disks")
    


    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank
    """
    
    """
    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """
    
    
    # plt.imshow(color_im)
    # plt.show()
    
    def filter_image(im, filters):
        return np.array([signal.convolve2d(im, f, 'same') for f in filters])
    
    # k-means clustering K=64 for filter responses at each pixel
    def create_texton_map(im, filters, k=64):
        # Step 1: Apply filters to the image
        responses = filter_image(im, filters)
        
        # Step 2: Reshape the responses to a 2D array where each row is a pixel, and each column is a filter response
        features = responses.reshape((responses.shape[0], -1)).T  # Ensure each row is a feature vector for a pixel
        print(responses.shape, features.shape)

        centroid, label = kmeans2(features, k, minit='points')
        texton = label.reshape(responses.shape[1:])
        
        return texton

    
    all_filters = DoG + LMS + LML + gabor
    
    texton_map = create_texton_map(im, all_filters)
    # plt.imshow(texton_map, cmap='viridis')
    # plt.show()
    cv2.imwrite(f'outputs/map/TextonMap_{im_idx}.png', texton_map)
    
    
    print("Done with texton map")

    """
    Generate Texton Gradient (Tg)
    Perform Chi-square calculation on Texton Map
    Display Tg and save image as Tg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    def chi_sq(im, bins, discs):
        dist = np.zeros_like(im, dtype=np.float64)
        l, r = discs
        for i in range(bins):
            tmp = np.float32(im == i)
            # print(im.shape, tmp.shape, l.shape)
            # convolve tmp with disc
            g = signal.convolve2d(tmp, l, 'same')
            h = signal.convolve2d(tmp, r, 'same')
            dist += .5 * (g - h) ** 2 / (g + h + np.float64(1e-6))
            
        return dist
    
    def create_gradient(map_, discs, k):
        gradient = []
        # (3, 16)
        for i in range(disc_size[0]):
            for j in range(disc_size[1] // 2): # discs should be even length orientation
                print(
                    i * disc_size[1] + j, 
                    i * disc_size[1] + j + disc_size[1] // 2
                )
                half_discs = discs[i * disc_size[1] + j], discs[i * disc_size[1] + j + disc_size[1] // 2]
                dist = chi_sq(map_, k, half_discs)
                gradient.append(dist)
        
        print(np.array(gradient).shape)
        return np.mean(gradient, axis=0)
    
    half_disc
    tg = create_gradient(texton_map, half_disc, 64)
    # plt.imshow(tg, cmap='viridis')
    # plt.show()
    plt.imsave(f'outputs/gradient/TextonGradient_{im_idx}.png', tg, cmap='viridis')
    
    
    print("Done with texton gradient")

    """
    Generate Brightness Map
    Perform brightness binning 
    """
    
    def create_brightness_map(im, k=16):
        features = im.reshape(-1, 1).astype(np.float32)
        print(im.shape, features.shape)
        
        centroid, label = kmeans2(features, k, minit='points')
        bright = label.reshape(im.shape)
        return bright
    
    brightness_map = create_brightness_map(im)
    # plt.imshow(brightness_map, cmap='viridis')
    # plt.show()


    """
    Generate Brightness Gradient (Bg)
    Perform Chi-square calculation on Brightness Map
    Display Bg and save image as Bg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    
    bg = create_gradient(brightness_map, half_disc, 16)
    # plt.imshow(bg, cmap='viridis')
    # plt.show()
    plt.imsave(f'outputs/gradient/BrightnessGradient_{im_idx}.png', bg, cmap='viridis')


    """
    Generate Color Map
    Perform color binning or clustering
    """
    
    # plt.imshow(color_im)
    
    def create_color_map(im, k=16):
        features = im.reshape(-1, 3).astype(np.float32)
        centroid, label = kmeans2(features, k, minit='points')
        color = label.reshape(im.shape[:2])
        return color
    
    color_map = create_color_map(color_im)
    # plt.imshow(color_map, cmap='viridis')
    # plt.show()
    

    """
    Generate Color Gradient (Cg)
    Perform Chi-square calculation on Color Map
    Display Cg and save image as Cg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    
    cg = create_gradient(color_map, half_disc, 16)
    # plt.imshow(cg, cmap='viridis')
    # plt.show()
    plt.imsave(f'outputs/gradient/ColorGradient_{im_idx}.png', cg, cmap='viridis')


    """
    Read Sobel Baseline
    use command "cv2.imread(...)"
    """
    sobel = cv2.imread(f'BSDS500/SobelBaseline/{im_idx}.png', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(sobel, cmap='viridis')
    # plt.show()

    """
    Read Canny Baseline
    use command "cv2.imread(...)"
    """
    canny = cv2.imread(f'BSDS500/CannyBaseline/{im_idx}.png', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(canny, cmap='viridis')
    # plt.show()


    """
    Combine responses to get pb-lite output
    Display PbLite and save image as PbLite_ImageName.png
    use command "cv2.imwrite(...)"
    """
    print(sobel.shape, canny.shape)
    
    
    baseline = (.5 * sobel + .5 * canny)
    cool_stuff = (tg + bg + cg) / 3
    
    print(baseline.shape, cool_stuff.shape)
    
    pb = cool_stuff * baseline
    
    plt.imshow(pb, cmap='viridis')
    plt.show()
    plt.imsave(f'outputs/color/PbLite_{im_idx}.png', pb, cmap='viridis')
    cv2.imwrite(f'outputs/grey/PbLite_{im_idx}.png', pb)
    

if __name__ == '__main__':
    for i in range(1, 11):
        main(i)