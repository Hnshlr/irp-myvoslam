# AY22 Individual Research Project (A) 2022-2023

Research Project on Autonomous Vehicle Self Awareness using SLAM algorithms.

This project is research as part of the Master Thesis for the MSc in Computational and Software Techniques at Cranfield University.

It aims to provide an analysis of the extent to which Visual Odometry can achieve accurate position and orientation estimation of a vehicle in complex environments.

© Copyright 2023, All rights reserved to Hans Haller, CSTE-CIDA Student at Cranfield Uni. SATM, Cranfield, UK.

‎ 

## Setting up the application

### Python version

The application was initially developed with Python 3.7 (3.7.17 to be exact), however in the late stages of development, a proper virtual environment was created under Python 3.9. In both cases, make sure you use respectively the ```requirements37.txt``` or ```requirements39.txt``` file to install the dependencies, depending on the version of Python you wish to use.

### Important notes

If Python 3.9 is used, you won't be able tu use ```SURF``` as a feature detector, as it is not supported by OpenCV 4.5.4, the lowest version of OpenCV that supports Python 3.9, as it was patented by its creators. If you wish to use ```SURF``` as a feature detector, please use Python 3.7.17.

### Initiate the virtual environment

Once the repository of the project is cloned, or downloaded from Cranfield's Canvas, create a virtual environment for the project to store all packages the app requires. 

To create a virtual environment, first open your command prompt or terminal and navigate to the main directory:

```cd ./irp-myvoslam```

Once in the main directory, create a new virtual environment using the venv module in Python. The command to create a virtual environment is:

```python3 -m venv venv```

This will create a new directory called "venv" in the backend directory, which will contain all the necessary files for their virtual environment. 

To activate the virtual environment, run the following command:

```source venv/bin/activate```

This will activate the virtual environment. You should see the name of the environment in your terminal prompt. Once the virtual environment is activated, you can install the dependencies required for the project using pip. The list of the required packages is listed in the requirements.txt file in the root of the directory backend.

To install the dependencies, run the following command:

```pip3 install -r requirements39.txt --no-cache-dir --no-dependencies``` (or ```pip3 install -r requirements37.txt --no-cache-dir --no-dependencies``` if you are using Python 3.7.17)

This will install all the required packages for the project. **Important note**: It is mandatory to add the ```--no-cache-dir``` and ```--no-dependencies``` flags to the command, otherwise the installation will fail. For further explanation, please refer to the end of the README.md file.

### Download the data

To download the data, run the following command:

```./scripts/curl-data-and-models.sh```

or 

```./scripts/wget-data-and-models.sh```

This will download the data from the KITTI dataset, create the ```src/data/input/kitti``` and ```src/data/output/kitti``` directories, and unzip the data in the input directory. It will also download the pre-trained models for the feature detectors and descriptors, and unzip them in the ```src/data/models``` directory. You must have either ```curl``` or ```wget``` installed on your machine for this to work.


### Run the application

Once the virtual environment is now set up and ready to use, the user can start the application by running the app.py file through their IDE, or using the following command:

```python3 main.py```

### You're set to go!

If you have any questions, please contact me at ```hans.haller.885@cranfield.ac.uk```

‎ 

## Running specific tasks

The application is designed to be modular, and to allow the user to run specific tasks. The following sections will describe how to run specific tasks.

### 1. Compute the pose estimation

To compute a pose estimation, simply run the following command:

```python3 main.py```

This will run the application with the default parameters, and will compute the pose estimation for the the sequences 4 and 5 of the KITTI dataset. The results will be saved in the ```src/data/output/kitti``` directory. Results are not saved by default.

There are various parameters that the user may tweak, and they are free to do so. The following sections will describe what can be done:

### 2. Tweaking Parameters

#### Dataset Paths:

The `datasets_paths` list contains paths to the datasets you want to test on. By default, it includes paths to all the datasets labeled from "S0" to "S10" in the `input_dir`.

```python
datasets_paths = [
    os.path.join(input_dir, dataset_index)
    for dataset_index in
    ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
]
```

To test on specific datasets, simply modify this list to include only the desired dataset labels.

#### Method:

The `method` parameter determines the type of visual odometry to be used (either "mono" or "stereo")

```python
method = ["mono", "stereo"][0]
```

The user may choose to run the application with either the monocular or stereo visual odometry method.

#### Feature Detection Parameters:

The `fd_parameters` dictionary contains parameters related to feature detection. You can set the feature detection method and the number of features to detect.

```python
fd_parameters = {
    "fda": ["surf", "fast", "orb"][2],   # The feature detection method
    "nfeatures": 3000                    # The number of features to detect
}
```

- `fda`: This parameter determines the feature detection algorithm to be used. In the given example, "orb" is selected.
- `nfeatures`: This parameter specifies the number of features to detect in the image.

By tweaking these parameters, users can experiment with different feature detection algorithms and the number of features to understand their impact on the visual odometry results.

#### Post Matching Outlier Removal (PMOR) Parameters

Post Matching Outlier Removal is crucial for refining the matches obtained after feature detection and description. Adjusting the parameters in the `pmor_parameters` dictionary allows you to control the outlier removal process:

```python
pmor_parameters = {
    "do_PMOR": True,
    "do_xyMeanDist": True,
    "do_xyImgDist": True,
    "do_RANSAC": True,
}
```

- `do_PMOR`: This is the main switch. Set to `True` to enable PMOR, and `False` to disable it.
- `do_xyMeanDist`: Toggle this to enable or disable the mean distance method for outlier removal.
- `do_xyImgDist`: Toggle this to enable or disable the image dimension method for outlier removal.
- `do_RANSAC`: Toggle this to enable or disable the RANSAC method for outlier removal.

#### Semantic Segmentation (SS) Parameters

Semantic Segmentation aids in understanding the scene by classifying each pixel into predefined categories. Adjust the parameters related to semantic segmentation in the `ss_parameters` dictionary:

```python
ss_parameters = {
    "do_SS": True,
    "model_path": "src/models/deeplabv3_xception65_ade20k.h5",
    "features_to_ignore": ["sky", "person", "car"]
}
```

- `do_SS`: Toggle to enable or disable semantic segmentation.
- `model_path`: Path to the pre-trained model used for semantic segmentation.
- `features_to_ignore`: A list of features or objects that you want excluded from the final mask applied to the image.

#### Frame Tile Optimization (FTO) Parameters

Frame Tile Optimization is used to distribute features uniformly across the image frame. Adjust the parameters in the `fto_parameters` dictionary:

```python
fto_parameters = {
    "do_FTO": True,
    "grid_h": 40,
    "grid_w": 20,
    "patch_max_features": 10
}
```

- `do_FTO`: This is the main switch. Set to `True` to enable FTO, and `False` to disable it.
- `grid_h` and `grid_w`: Define the grid size for the image frame. This determines how many tiles the image is divided into, horizontally and vertically.
- `patch_max_features`: The maximum number of features you want to retain in each tile. This ensures a uniform distribution of features across the frame.

#### Visualisation, Monitoring, and Saving

1. **Visualization (`view` parameter, set to `True` by default)**: Enables real-time visualization of the visual odometry process, showcasing keypoints detection, frame matching, and motion estimation.
   
2. **Monitoring (`monitor` parameter, set to `True` by default)**: Uses the `tqdm` library to display a progress bar, offering a real-time status of the computation's progress.

3. **Saving (`save` parameter, set to `False` by default)**: Allows the results, specifically the Absolute Trajectory Error (ATE), to be saved into a `.csv` file for further analysis.

Feel free to adjust these parameters as you see fit.

### 3. Bulk-Testing of FTO grid combinations

The application allows the user to run a bulk test of FTO grid combinations. This is useful to determine the optimal grid size for the image frame. To run a bulk test, simply run the following command:

```python3 bulk.py```

#### Parameters Overview:

1. **GRID_H_values & GRID_W_values**: These lists define the various grid sizes you want to test. For instance, `GRID_H_values = [4, 8, 10]` means you'll test with 4, 8, and 10 tiles in the horizontal direction.

2. **PATCH_MAX_FEATURES**: This parameter ensures the number of features never exceeds a certain threshold. It's set to 10 by default, meaning each tile will have a maximum of 10 features.

3. **datasets_paths**: This list contains paths to the datasets you want to test on.

4. **fd_parameters**: This dictionary contains parameters related to feature detection. You can set the feature detection method and the number of features to detect.

#### Running with Custom Parameters:

If you wish to test with different parameters, modify the appropriate variables in the `bulk.py` script. For instance, to test with different grid sizes, simply modify the `GRID_H_values` and `GRID_W_values` lists.

After adjusting the parameters as desired, run the script again:

```bash
python3 bulk.py
```

#### Results:

The results will be saved in the `src/data/output/kitti` directory. Three files will be generated:

1. **BULKFTO_[timestamp].csv**: Contains the raw results for each dataset and grid combination.
2. **BULKFTO_ATE_COMPARISON_[timestamp].csv**: Contains the Absolute Trajectory Error (ATE) comparison.
3. **BULKFTO_NCATE_COMPARISON_[timestamp].csv**: Contains the Normalized Cumulative ATE comparison.


‎ 

## Contact

For further assistance or inquiries, don't hesitate to reach out to me at `hans.haller.885@cranfield.ac.uk`, or create a new issue on the repository.

Happy experimenting!
