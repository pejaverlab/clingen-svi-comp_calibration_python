This is a python version of [clingen-svi-comp_calibration](https://github.com/vpejaver/clingen-svi-comp_calibration)

We have tested this tool with python 3.6 and python 3.8.
The tool relies on numpy and matplotlib and should work with recently fair version of these libraries.

To run the code:
```
python main.py --configfile "$PATH_TO_CONFIG_FILE" --tool="$TOOL_NAME" --labelled_data_file "$PATH_TO_LABELLED_DATA_FILE" --unlabelled_data_file "$PATH_TO_UNBALELLED_DATA_FILE" --outdir="out"
```
The results are stored in "out" sub directory.



Description of Tuning Parameters defined in config.ini are as follows:
```python

[tuningparameters]
B = 1000     # Number of Bootstrap Iterations for computing the Discounted Thresholds
discountonesided = 0.05      # While computing thresholds,
windowclinvarpoints = 100    # For the adaptive windows for computing the local probabilty, this defines the minimum number of 'labelled data points' that should be in the window 

[priorinfo]
emulate_tavtigian = False   # Use 'c' and 'alpha' as in Tavtigian et al
emulate_pejaver = False     # Use 'c' and 'alpha' as in Pejaver et al
alpha = 0.0441              # define alpha by yourself and compute 'c' as per Tavtigian et al framework

[smoothing] 
gaussian_smoothing = False   # Apply Gaussian Smoothing on result
data_smoothing = True       # Set True of Unbalelled Data is available and to be used for smoothing
windowgnomadfraction = 0.03  # For the adaptive	windows for computing the local  probabilty, this defines the minimum fraction of 'unlabelled data points' that should be in the window

```



An example use of invoking Local Calibration Method. Refer examples/example2.py for the whole code


```python
    calib = LocalCalibration(alpha, c, reverse, clamp, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing)
    thresholds, posteriors_p = calib.fit(x,y,g,alpha)

```


An example of using Local Calibration Method to compute BootStrapped Discounted Thresholds. Refer examples/example3.py for the whole code


```python
    calib = LocalCalibrateThresholdComputation(alpha, c, reverse, clamp, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing, )
    _, posteriors_p_bootstrap = calib.get_both_bootstrapped_posteriors_parallel(x,y, g, 1000, alpha, thresholds)

    Post_p, Post_b = get_tavtigian_thresholds(c, alpha)

    all_pathogenic = posteriors_p_bootstrap
    all_benign = 1 - np.flip(all_pathogenic, axis = 1)

    pthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_pathogenic, thresholds, Post_p)
    bthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_benign, np.flip(thresholds), Post_b) 

    DiscountedThresholdP = LocalCalibrateThresholdComputation.get_discounted_thresholds(pthresh, Post_p, B, discountonesided, 'pathogenic')
    DiscountedThresholdB = LocalCalibrateThresholdComputation.get_discounted_thresholds(bthresh, Post_b, B, discountonesided, 'benign')


```

You can run the example as modules:

```
python -m examples.example1 --configfile "$PATH_TO_CONFIG_FILE" --tool="$TOOL_NAME" --labelled_data_file "$PATH_TO_LABELLED_DATA_FILE" --unlabelled_data_file "$PATH_TO_UNBALELLED_DATA_FILE" --outdir="out"
python -m examples.example2 --configfile "$PATH_TO_CONFIG_FILE" --tool="$TOOL_NAME" --labelled_data_file "$PATH_TO_LABELLED_DATA_FILE" --unlabelled_data_file "$PATH_TO_UNBALELLED_DATA_FILE" --outdir="out"
python -m examples.example3 --configfile "$PATH_TO_CONFIG_FILE" --tool="$TOOL_NAME" --labelled_data_file "$PATH_TO_LABELLED_DATA_FILE" --unlabelled_data_file "$PATH_TO_UNBALELLED_DATA_FILE" --outdir="out"
```