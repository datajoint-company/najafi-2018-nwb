# najafi-2018-nwb

This project presents the data accompanying the paper
> Najafi, Farzaneh, Gamaleldin F. Elsayed, Eftychios Pnevmatikakis, John Cunningham, and Anne K. Churchland. "Inhibitory and excitatory populations in parietal cortex are equally selective for decision outcome in both novices and experts." bioRxiv (2018): 354340.

https://www.biorxiv.org/content/early/2018/10/10/354340

The original data are available from Cold Spring Harbor Laboratory:  http://repository.cshl.edu/36980/

## Conversion to NWB 2.0 file
This repository will contain the Python 3+ code to convert the dataset into the NWB 2.0 format (See https://nwb.org)


To start, download the dataset from http://repository.cshl.edu/36980/, follow the instruction to concatenate and extract data.
The resulted data directory includes a **manifest.txt** file specifying all available data, and a data folder containing the *".mat"* files

The conversion to NWB 2.0 format is done via the [**NWB_conversion.py**](https://github.com/ttngu207/najafi-2018-nwb/blob/master/scripts/NWB_conversion.py) script. This script takes one argument, a *.json* config file, specifying the *manifest* file, output directory, and some metadata. 

An example content of the *.json* config file is as follow: 
```
{
	"manifest": "data/manifest-md5.txt",
	"general": 
		{
			"experimenter" : "Farzaneh Najafi",
			"institution" : "Cold Spring Harbor Laboratory",
			"related_publications" : "https://doi.org/10.1101/354340"
		},
	"error_log" : "data/conversion_error_log.txt",
	"output_dir" : "data/NWB 2.0"
}
```

The converted NWB 2.0 files will be saved in the *output_dir* directory specified in the *.json* file. Running the conversion script is as follow: 
```
python NWB_conversion conversion_config.json

```
## Showcase work with NWB:N files
This repository will contain Jupyter Notebook demonstrating how to navigate and query the dataset. 

See this [Jupyter Notebook](https://github.com/ttngu207/najafi-2018-nwb/blob/master/notebooks/NWB2.0_Tutorial.ipynb) for a tutorial on using [**PyNWB**](https://pynwb.readthedocs.io/en/latest/) API to access NWB 2.0 data, to process and plot some of the key figures presented in this study (https://doi.org/10.1101/354340). 
