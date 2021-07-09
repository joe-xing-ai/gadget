
## Project Gadget

Gadget is an Image Embedding Based Search & Data-centric AI Exploration Engine with the primary focus
being proactively explore and look for Smart Data that makes your DNN model stands out.

Image embedding vectors are used to build index around image data for a quick search based upon similart
metrics, etc.

#### Author: joe.xing.ai@gmail.com

### Design

The model

The system


### Setup

- conda 4.10.1
- conda env create --file environment_ubuntu.yaml (environment_windows.yml)
    for now assumes we only supports Ubuntu and Windows 10
- conda activate gadget
- cd ./python/ && python main.py --download --dataset_name "food01" --image_folder "./data/food01/"
    these initialization procedures will start to download Tensorflow standard datasets.

### QA System - Sanity Checks, Qualitative & Quantitative Measurements of the Model Performance

#### (a) Distribution of Image Embedding Vectors

We should always visualize how the Image Embedding Vectors distribute in Euclidean Space after the dimensionality
reduction. The figure shows a sanity check for distributions of embedding vectors for 3 different classes of images:
images that have car, food and dogs within the ROI. The embedding vectors are nicely separated and showing clustering 
behaviors in phase space. We take 3 classes of image data from the Tensorflow standard sample dataset, "cars196",
"food101" and "stanford_dogs".

<p align="center">
  <img src="python/artifacts/embedding_distribution.png" width="1000" title="Distribution of Image Embedding Vectors">
</p>


#### (b) Image Search Results

### Deployment

