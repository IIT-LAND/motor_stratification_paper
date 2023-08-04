# motor_stratification_paper

In this repository you will find the pipeline to reproduce the anlysis of the "motor stratification paper" (change name when published)

There are 3 folders:
### MotorStratificationModel
to reproduce the stratification model implementation. It contatins 4 main scripts:
- **00_data_exploration.Rmd**: 1st scipt to run to merge IRCCS MEDEA and NDA dataset, run some preliminary statistical analysis to test the different motor performance in Autism with respect to Typically Developing children and children with DCD
- **01_preprocessing.Rmd**: script that does beta correction for removing variance in the data given by recruiting site and MABC2 module administered
- **02_reval4ABC.py**: scritp that run reval (cit) for build the autism motor stratification model. this script need to be runned into a specific python evvironment, whose characteristics are available in the *requirements_mot.txt*. 
- **03_comparing_clustering.Rmd**: this script produce visulization and statistical description for the autism motor clusters

### KinematicAnalysis
it contains script to do the kinematic analysis of the reach-to-drop motor task, starting from data recorded. with an optoelettronic available for the IRCCS-MEDEA dataset:
- **A_RawKinematics2CSV.m** : this 1st script writes a .csv files for each recordered trial from the raw tracked-data (.tdf)
- **B_CutTimeSeries_startend.Rmd**: this 2nd script cut the from the timeseries of each trial the recording of what happen before and after the real movement (children often start to move few second after the recording started)
- **B01_segmentatation_definepeaks.Rmd** and **B02_segmentation_cutsegments.Rmd**: - this 2 scripts respectively identifies the deceleration peaks for each trial's reach movement and segment the time series into the feedforward and feedback phases
- **C_DTW_reach2drop.Rmd**: this script runs the multivariate Dynamic Time Warping (DTW) for the entire reach-to-drop 
- **C01_DTW_reach2drop_segmentation.Rmd**: this script runs the DTW for the feedforward and feedback phases separately
- **D_DTW_results_exploration.Rmd**: this script runs visualization and statistics for the entire movement's motor noise
- **D01_DTW_results_exploration_segmentation.Rmd**:this script runs visualization and statistics for feedforward and feedback phases' motor noise

### GeneExpressionAnalysis
it contains the code for running the gene expression analysis.
- **_gex_decoding.Rmd**: this script runs the gene decoding analysis and produce visualizations.
- **genelistOverlap.R**: the script calculates enrichment odds ratio and p-value from hypergeometric test to answer the question of whether genes from one list are enriched in genes from another list
