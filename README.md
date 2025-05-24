# BCCSS
Bayesian Classifier Calibration Based on Sythesized Samples for Zero-shot Chinese Character Recognition

## Environment
python 3.8  
pytorch 1.8

## Data
HWDB: [link]()  
CTW: [link]()  
Auxiliary Printed Data: `printed.hdf5` and `printed_ctw3650.hdf5`


## Training

### Classifier
For HWDB, please run the script
```
bash ./hwdb/run_hwdb.sh
```
<!-- ### CTW
Run the shell script
```
bash ./hwdb/run_ctw.sh
``` -->

### Generator
The code of generator is based on [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).   
For quick reimplementation of the method, you can download the provided [synthesized data]() of unseen characters in the experiment HWDB-500. 


## Calibration and Test

For HWDB, please run the script
```
bash ./hwdb/run_calibration.sh
```
<!-- ### CTW
Run the shell script
```
bash ./hwdb/run_calibration_effect.sh
``` -->
