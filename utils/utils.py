
def get_smaller_grid_size(number_images):
    '''
    Calculates the smallest grid size for any given number of images.
    
    Args:
        number_images (int): number of images
        
    Return:
        (image batch, label batch): Image and label batch
    '''
    assert number_images > 0
    
    out = ceil(sqrt(number_images))
    return  out, out


def plot_grid(ims, interp=False, titles=None):
    
    '''
    
    Given an array with images 'ims', this function plot every image in a grid.

    Args:
        ims (array): array of images
        interp (string): interpolation method, see: http://matplotlib.org/1.4.3/examples/images_contours_and_fields/interpolation_methods.html
        titles (string): Titles for the plots (eg.: the classes of each image)
        
    Return:
        (image batch, label batch): Image and label batch
        
    
    '''
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3): # is not a channel last image?
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=(15,10))
    f.subplots_adjust(wspace=0.02,hspace=0)
    
    width, height = get_smaller_grid_size(len(ims))
    for i in range(len(ims)):
        sp = f.add_subplot(width, height, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.title.set_text(titles[i])
        plt.imshow(ims[i], interpolation=None if interp else 'none')