import cupy as cp

def ft2d(image):
    ft = cp.fft.fft2(image)
    return cp.fft.fftshift(ft)
def ift2d(image):
    ift = cp.fft.ifftshift(image)
    ift = cp.fft.ifft2(ift)
    return ift.real

def normalize(image):
    return (image - cp.min(image)) / (cp.max(image) - cp.min(image))

def low_pass_filter_gaussian(image, cutoff_angstrom, pixel_size):
    '''
    low pass filter for the image using Gaussian filter.
    Args:
        image: image
        cutoff_angstrom: cutoff angstrom
        pixel_size: pixel size (angstrom)
    Returns:
        ift: low pass filtered image
    '''
    rows, cols = image.shape
    freq_x = cp.linspace(-0.5, 0.5, cols)
    freq_y = cp.linspace(-0.5, 0.5, rows)
    fx, fy = cp.meshgrid(freq_x, freq_y)
    freq_radius = cp.sqrt(fx**2 + fy**2)
    
    cutoff_frequency = 2 * pixel_size / cutoff_angstrom
    mask = cp.exp(-(freq_radius**2) / (2 * (cutoff_frequency)**2))

    image = normalize(image)

    f_shifted = ft2d(image)
    f_filtered_shifted = f_shifted * mask
    ift = ift2d(f_filtered_shifted)

    ift = normalize(ift)
    return ift.get()