This is a python version of [clingen-svi-comp_calibration](https://github.com/vpejaver/clingen-svi-comp_calibration)

We have tested this tool with python 3.6 and python 3.8.
The tool relies on numpy, scipy and matplotlib and should work with fairly recent version of these libraries.


The tool offers two modes:

1. Calibrate
2. Infer


## Calibrate

To run the tool in Calibrate mode:
```
python main.py --configfile config.ini --labelled_data_file "$PATH_TO_LABELLED_DATA_FILE" --unlabelled_data_file "$PATH_TO_UNBALELLED_DATA_FILE" --outdir "$PATH_TO_RESULT_DIR"
```
The labelled_data_file should be a two column tab separated file where first column is the score and second column is the label (0 for benign and 1 for pathogenic).

The unlabelled_data_file should can be a single or double column tab separated file where first column stores the scores.

The scores by default are assumed to be positively correlated with pathogenicity. If the scores are negatively correlated with pathogenicity, add "--reverse" to the passed argument like:
```
python main.py --configfile config.ini --labelled_data_file "$PATH_TO_LABELLED_DATA_FILE" --unlabelled_data_file "$PATH_TO_UNBALELLED_DATA_FILE" --outdir "$PATH_TO_RESULT_DIR" --reverse
``` 

The results are stored in "$outdir" sub directory.


#### Tuning Parameters

Description of Tuning Parameters defined in config.ini are as follows:
```python

[tuningparameters]
B = 10000     # Number of Bootstrap Iterations for computing the Discounted Thresholds
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


#### Results

In the provided output directory, you will have following files:
1. $TOOL_NAME_benign.txt   - This is a two column file. The first column represents a score from the tool and second column represents the calibrated probability of being benign.
2. $TOOL_NAME_pathogenic.txt - This is a two column file.   The first column represents a score from the tool and second column represents the calibrated probability of being pathogenic.
3. $TOOL_NAME_bthresh.txt - This is a five column file. It gives a score threshold for five levels of evidence strength for benignity. Each row corresponds to a bootstrap iteration. (excpet first row which includes whole data)
4. $TOOL_NAME_pthresh.txt - This is a five column file.	It gives a score threshold for five levels of evidence strength	for pathogenicity. Each row corresponds to a bootstrap iteration. (excpet first row	which includes whole data)
5. $TOOL_NAME_bthreshdiscounted.txt - This reports 95 percentile score for each evidence strength for benignity computed from all bootstrap iteration
6. $TOOL_NAME_pthreshdiscounted.txt- This reports 95 percentile score for each evidence strength for pathogenicity computed	from all bootstrap iteration
7. $TOOL_NAME_benign.png - This is a score-probability graph for benignity.
8. $TOOL_NAME_pathogenic.png - This  is a score-probability graph for pathogenicity.




#### Additional Comments

In case you wish to use our implementation of  Local Calibration method in your own tool, refer to following hints:


1. An example use of invoking Local Calibration Method. Refer examples/example2.py for the whole code


```python
    calib = LocalCalibration(alpha, c, reverse, clamp, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing)
    thresholds, posteriors_p = calib.fit(x,y,g,alpha)

```


2. An example of using Local Calibration Method to compute BootStrapped Discounted Thresholds. Refer examples/example3.py for the whole code


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



## Infer

The 'infer' functionality is to get the pathogenicity classification of a variant.

To run infer, there are two options:

1. Calibration as per Pejaver et al.

In this case, we provide the score/score file and the tool name :
```
python main.py infer --score "$SCORE" --tool_name BayesDel-noAF
```
or provide the score file, which just contains list of scores
```
python main.py infer --score_file "$PATH TO SCORE FILE"  --tool_name BayesDel-noAF
```

2. Use your own calbrated data directory generated from the "calibrate" option above:
```
python main.py infer --score "$SCORE" --calibrated_data_directory "$PATH TO CALIBRATED DATA DIRECTORY"
```

or provide the score file:
```
python main.py infer --score_file "$PATH TO SCORE FILE" --calibrated_data_directory "$PATH TO CALIBRATED DATA DIRECTORY"
```

