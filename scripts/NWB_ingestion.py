import os
from datetime import datetime
from dateutil.tz import tzlocal
import pytz
import re


import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pynwb
from pynwb import NWBFile, NWBHDF5IO
from pynwb import ophys as nwb_ophys


# Load data - mouse 1 _ fni16

path = os.path.join('data', 'data', 'mouse1_fni16', '150817')
more_fname = 'more_150817_001_ch2-PnevPanResults-170808-190057.mat'
post_fname = 'post_150817_001_ch2-PnevPanResults-170808-190057.mat'

moremat = sio.loadmat(os.path.join(path, more_fname), struct_as_record = False, squeeze_me = True)
postmat = sio.loadmat(os.path.join(path, post_fname), struct_as_record = False, squeeze_me = True)

alldata_frameTimes = postmat['alldata_frameTimes']


# NWB 2.0
# step 1: ingest metadata (hard-coded for now ...)

# -- NWB file - a NWB2.0 file for each session
datetime_format_yymmdd = '%y%m%d'
session_start_time = datetime.strptime('150817', datetime_format_yymmdd)
session_start_time.astimezone(pytz.timezone('US/Eastern'))  # assuming the recording is done at NY, NY

nwbfile = NWBFile(
    session_description = '150817_001_ch2-PnevPanResults-170808-190057',
    identifier = '150817_001_ch2-PnevPanResults-170808-190057.mat',
    session_id = '150817_001_ch2-PnevPanResults-170808-190057',
    session_start_time = session_start_time,
    file_create_date = datetime.now(tzlocal()),
    experimenter = 'Farzaneh Najafi',
    institution = 'Cold Spring Harbor Laboratory',
    related_publications = 'https://doi.org/10.1101/354340'
)

# -- imaging plane - the plane info ophys was performed on (again hard-coded here)
device = pynwb.device.Device('imaging_device_1')
nwbfile.add_device(device)
optical_channel = nwb_ophys.OpticalChannel('my_optchan', 'description', 500.)
imaging_plane = nwbfile.create_imaging_plane(
    name = 'img_pln',
    optical_channel = optical_channel,
    device = device,
    description = 'imaging plane 123 ',
    excitation_lambda = 123.0,
    imaging_rate = 123.0,
    indicator = 'GFP',
    location = 'brain loc #1',
    manifold = np.ones((5, 5, 3)),
    conversion = 1.23,
    unit = 'meter'
)

# -- image segmentation

img_segmentation = nwb_ophys.ImageSegmentation(name='img_seg')
nwbfile.add_processing_module([img_segmentation])

plane_segmentation = nwb_ophys.PlaneSegmentation(
    name = 'pln_seg',
    description = 'description here',
    imaging_plane = imaging_plane
)
img_segmentation.add_plane_segmentation([plane_segmentation])

# add more columns
plane_segmentation.add_column(name = 'roi_id', description = 'roi id')
plane_segmentation.add_column(name = 'roi_status', description = 'good or bad ROI')
plane_segmentation.add_column(name = 'fitness', description = '')
plane_segmentation.add_column(name = 'neuron_type', description = 'excitatory or inhibitory')
plane_segmentation.add_column(name = 'roi2surr_sig', description = '')
plane_segmentation.add_column(name = 'offsets_ch1_pix', description = '')

# start inserting ROI mask
tmp = np.empty(moremat['idx_components'].shape)
tmp.fill(np.nan)
neuron_type = tmp
neuron_type[np.where(moremat['badROIs01'] == 0)] = moremat['inhibitRois_pix']
roi2surr_sig = tmp
roi2surr_sig[np.where(moremat['badROIs01'] == 0)] = moremat['roi2surr_sig']
offsets_ch1_pix = tmp
offsets_ch1_pix[np.where(moremat['badROIs01'] == 0)] = moremat['offsets_ch1_pix']

for idx, idval in enumerate(moremat['idx_components']):
    plane_segmentation.add_roi(
        roi_id = idval,
        image_mask = moremat['mask'][:, :, idx],
        roi_status = 'good' if moremat['badROIs01'][idx] == 0 else 'bad',
        fitness = moremat['fitness'][idx],
        neuron_type = 'inhibitory' if neuron_type[idx] == 1 else 'excitatory' if neuron_type[idx] == 0 else 'unknown',
        roi2surr_sig = roi2surr_sig[idx],
        offsets_ch1_pix = offsets_ch1_pix[idx]
    )

# create a ROI region table
roi_region = plane_segmentation.create_roi_table_region(
    name = 'good_roi',
    description = 'good roi region table',
    region = (np.where(moremat['badROIs01'] == 0)[0]).tolist()
)
# roi_region = nwb_ophys.DynamicTableRegion(
#     name = 'good_roi',
#     description = 'good roi region table',
#     data = (np.where(moremat['badROIs01'] == 0)[0]).tolist(),
#     table = plane_segmentation
# )
# create an "analysis" group
dF_F = nwb_ophys.DfOverF(name='dF over F')
nwbfile.add_analysis(dF_F)

# now build "RoiResponseSeries" by ingesting data
# let's just do this for trial #1 for now
def build_roi_series(data_string_name, post_data, dyn_table):
    trial_num = 1
    roi_resp_series = nwb_ophys.RoiResponseSeries(
        name = data_string_name,
        data = post_data[data_string_name].traces.transpose([1, 0, 2]),
        #data = post_data[data_string_name].traces.transpose([1, 0, 2]),
        unit = '',
        rois = dyn_table,
        timestamps = post_data[data_string_name].time,
        description = 'ROIs x time x trial'
    )
    return roi_resp_series

dF_F.add_roi_response_series([
    build_roi_series('firstSideTryAl',postmat,roi_region),
    build_roi_series('firstSideTryAl_COM',postmat,roi_region),
    build_roi_series('goToneAl',postmat,roi_region),
    build_roi_series('rewardAl',postmat,roi_region),
    build_roi_series('commitIncorrAl',postmat,roi_region),
    build_roi_series('initToneAl',postmat,roi_region),
    build_roi_series('stimAl_allTrs',postmat,roi_region),
    build_roi_series('stimAl_noEarlyDec',postmat,roi_region),
    build_roi_series('stimOffAl',postmat,roi_region)
])

# -- Write NWB2.0 file
if False:
    save_path = os.path.join('data', 'nwb2.0')
    save_file_name = nwbfile.session_id
    with NWBHDF5IO(os.path.join(save_path, save_file_name), mode = 'w') as io:
        io.write(nwbfile)

    with NWBHDF5IO(os.path.join(save_path, save_file_name), mode = 'r') as io:
        nwbfile_in = io.read()







