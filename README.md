# neurodemo

Data analysis demos made initially for the 2015 Neurosciences bootcamp at UCSD. Code written and data collected by numerous students.

## fmri

Data collected from 3 participants, and voxels extracted from area V1.
Subjects participated in 8 blocks (36 trials each) of a task in which they were presented two flickering
oriented square wave gratings and asked to attend to one of the two gratings, as indicated by the attention cue
(small bar near fixation point) while keeping their eyes at fixation. During the trial (3 s duration),
the attended grating would change spatial frequency and participants responded as to whether spatial frequency
was briefly higher or lower. 

## CaImaging

Data collected from two mice free to run on a wheel at will. Visual stimulus presented for 4 seconds, followed
by an 8-second ISI. 12 oriented gratings are presented in random order in 10 sets for each mouse. Regions of
interest (ROIs) were extracted from the imaging, and inhibitory cells were labelled.

## Usage

Download this repo and add it to your MATLAB path.

When running the analysis, navigate to the directory where the data is stored.

### Dependencies

Users must have the Statistics Toolbox in order to run the decoding analysis.