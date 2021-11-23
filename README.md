

# BOUNDARY SEGMENTATION
---
### About

Superpixel based SNIC Implementation for image segmentation and boundary delineation. This is a wrapper for Simple Non-Iterative Clustering algorithm. Use this for segementing different areas of image in unsupervised manner, based on spectral differences

### Parameters

- ##### rasterio_image_object : rasterio object
    Rasterio object is required for geo-referencing of the output. Just pass in `rasterio.open()` here
        
- ##### seed : int, eg: 400
    Also known as `number of segments` in your image. Larger values give more closed polygons, there is no optimal value. Eg: If you give `100`, 
    it will try to create 100 closed polygons
   
- ##### output_file_name: string, eg: 'example.geojson'
    Specify the filename/full path where you want to output the geojson file

### Returns :
- #### Function returns:
    1. File Name
    2. GeoJson File

### DETECTING BOUNDARIES

**Superpixel**: An un-supervised algorithm that works by planting "seeds" across the image, which is required as an input. The idea is to cluster areas by pixel similarity of neighbors of the seeds. Can be very sensitive to pixel changes. The seed values is directly proportional to the number of areas to segment. 
 - High seed values will give you more number of closed polygons
 - High compactness values will give more uniform boundaries, more squared, but will pick up less variations
 
### VISUAL RESULTS
- Original Image

- SuperPixel Boundary


### INSTALLATION

```
pip install git+https://gitlab.com/
```

### EXTERNAL DEPENDENCIES

```
gdal >= 3.1

Input image should have at least 12 Bands (Sentinel - 2A) 
```

### HOW TO RUN

```
from training.unsupervised.segmentation import Segmentation
import rasterio

obj = Segmentation(rasterio.open('data/myimage.tif'), 400, 'sample.geojson')
obj.exportGeoJsonFile()
```

Check out the example file [here]

Download sample sentinel file [here](https://drive.google.com/file/d/1rbIGhc_968QoAB60leXqkEXbEfgzYROU/view?usp=sharing)