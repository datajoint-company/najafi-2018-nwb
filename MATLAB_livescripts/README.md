# najafi-2018-nwb

This project presents the data accompanying the paper
> Farzaneh Najafi, Gamaleldin F Elsayed, Robin Cao, Eftychios Pnevmatikakis, Peter E Latham, John Cunningham, Anne K Churchland. "Excitatory and inhibitory subnetworks are equally selective during decision-making and emerge simultaneously during learning" bioRxiv (2018): 354340.

https://doi.org/10.1101/354340

The original data are available from Cold Spring Harbor Laboratory:  http://repository.cshl.edu/36980/

# MATLAB Live Script
This project demonstrates the use of MATLAB and MATLAB Live Script in lastest Neuroscience studies. Specifically, the project provides guidance and examples for working with neurodata in the [Neurodata Without Border](https://neurodatawithoutborders.github.io) format (NWB 2.0)

Data are first downloaded and converted to NWB 2.0 format, using Python scripts, see description below.

The [Live Script](https://github.com/ttngu207/najafi-2018-nwb/blob/master/MATLAB_livescripts/najafi_examples_with_matnwb.mlx) provides examples of data querying, conditioning and visualization, with several figures from the paper reproduced (see [pdf](https://github.com/ttngu207/najafi-2018-nwb/blob/master/MATLAB_livescripts/najafi_examples_with_matnwb.pdf)). 


# Converting the original data
The data download instructions are for a Unix-family OS such as Linux or Mac OS with Python 3.7+ on the system path as `python3`. 

## Clone this repository and download the data
In the terminal window, git clone

```console
$ git clone https://github.com/vathes/najafi-2018-nwb.git
$ cd najafi-2018-nwb
``` 

## Download the original data 

The following command will download the original data from CSHL (~70 GB).
```console 
$ mkdir data
$ python3 scripts/download.py
```
This may take several hours.  If the download is interrupted, simply re-run `download.py` and it will pick up where it left.

Verify that all 18 files have downloaded.
```console
$ ls data
FN_dataSharing.tgz-aa	FN_dataSharing.tgz-af	FN_dataSharing.tgz-ak	FN_dataSharing.tgz-ap
FN_dataSharing.tgz-ab	FN_dataSharing.tgz-ag	FN_dataSharing.tgz-al	FN_dataSharing.tgz-aq
FN_dataSharing.tgz-ac	FN_dataSharing.tgz-ah	FN_dataSharing.tgz-am	FN_dataSharing.tgz-ar
FN_dataSharing.tgz-ad	FN_dataSharing.tgz-ai	FN_dataSharing.tgz-an
FN_dataSharing.tgz-ae	FN_dataSharing.tgz-aj	FN_dataSharing.tgz-ao
```

Now unpack the tar files:

```console
$ cat data/FN_dataSharing.tgz-a* | tar -C data -xzf -
```

Verify that the data have unpacked:

```console
$ ls data/FN_dataSharing
bag-info.txt		data			manifest-sha256.txt	tagmanifest-sha256.txt
bagit.txt		manifest-md5.txt	tagmanifest-md5.txt

$ ls data/FN_dataSharing/data
metaData  metaData~  mouse1_fni16  mouse2_fni17  mouse3_fni18  mouse4_fni19
```

The `FN_dataSharing` data directory includes a `manifest.txt` file specifying all available data, and a data folder containing the `.mat` files.


## Conversion to NWB 2.0
The following command will convert the dataset into the NWB 2.0 format (See https://neurodatawithoutborders.github.io/)

```console
$ mkdir data/nwb
$ python3 scripts/convert_to_nwb.py
```

The `convert_to_nwb` uses the configuration file `conversion_config.json` to specify the *manifest* file, the output file, and general data about the experiments.

An example content of the *.json* config file is as follow: 
```json
{
	"manifest": "data/manifest-md5.txt",
	"general": 
		{
			"experimenter" : "Farzaneh Najafi",
			"institution" : "Cold Spring Harbor Laboratory",
			"related_publications" : "https://doi.org/10.1101/354340"
		},
	"output_dir" : "data/nwb"
}
```

The converted NWB files will be saved in the `output_dir` directory. 

