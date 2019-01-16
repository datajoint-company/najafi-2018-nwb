# najafi-2018-nwb

This project presents the data accompanying the paper
> Najafi, Farzaneh, Gamaleldin F. Elsayed, Eftychios Pnevmatikakis, John Cunningham, and Anne K. Churchland. "Inhibitory and excitatory populations in parietal cortex are equally selective for decision outcome in both novices and experts." bioRxiv (2018): 354340.

https://www.biorxiv.org/content/early/2018/10/10/354340

The original data are available from Cold Spring Harbor Laboratory:  http://repository.cshl.edu/36980/

# Converting the original data
The data download instructions are for a Unix-family OS such as Linux or Mac OS with Python 3.7+ on the system path as `python3`. 

## Clone this repository and download the data
In the terminal window, git clone

```console
$ git clone https://github.com/vathes/najafi-2018-nwb.git
$ cd najafi-2018-nwb
$ mkdir data
``` 

The following script will download the raw data from CSHL (~70 TB) -- it may take several hours.

## Download the raw data 

```console 
$ python3 scripts/download.py
```

## Conversion to NWB 2.0
The following command will convert the dataset into the NWB 2.0 format (See https://neurodatawithoutborders.github.io/)

```console
$ python3 scripts/NWB_convert.py
```


## Showcase work with the dataset in the NWB 2.0 format from Python
This repository will contain Jupyter Notebooks show how to navigate and query the dataset. 

## Showcase work with the dataset in the NWB 2.0 format from MATLAB
This repository will contain Matlab scripts to show how to navigate and query the dataset. 
