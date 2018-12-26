import os
from datetime import datetime
from dateutil.tz import tzlocal
import pytz
import re

import numpy as np
import scipy.io as sio
import pynwb
from pynwb import NWBFile, NWBHDF5IO
from pynwb import ophys as nwb_ophys


# Setup some metadata information
datetime_format_yymmdd = '%y%m%d'
timezone = pytz.timezone('US/Eastern')  # assuming the recording is done at NY, NY

experimenter = 'Farzaneh Najafi'
institution = 'Cold Spring Harbor Laboratory'
related_publications = 'https://doi.org/10.1101/354340'

# Load data
data_path = os.path.join('data', 'data')
mouse_dirs = [n for n in os.listdir(data_path) if
              os.path.isdir(os.path.join(data_path, n))]
# Looping through each mouse
for mouse_dir in mouse_dirs:
    sess_dirs = [n for n in os.listdir(os.path.join(data_path, mouse_dir)) if
                 os.path.isdir(os.path.join(data_path, mouse_dir, n))]
    # Looping through all session
    for sess_dir in sess_dirs:
        fnames = os.listdir(os.path.join(data_path, mouse_dir, sess_dir))

        for fname in fnames:
            if re.match('more', fname):
                moremat = sio.loadmat(os.path.join(data_path, mouse_dir, sess_dir, fname),
                                      struct_as_record = False, squeeze_me = True)
            if re.match('post', fname):
                postmat = sio.loadmat(os.path.join(data_path, mouse_dir, sess_dir, fname),
                                      struct_as_record = False, squeeze_me = True)

        # NWB 2.0
        # step 1: ingest metadata (hard-coded for now ...)
        file_name = re.sub('more_|post_|.mat', '', fnames[0])

        # -- NWB file - a NWB2.0 file for each session
        session_start_time = datetime.strptime(sess_dir, datetime_format_yymmdd)
        session_start_time.astimezone(timezone)
        session_id = file_name.split('-')[-1]  # assuming the last 6-digits is the session id
        nwbfile = NWBFile(
            session_description = file_name,
            identifier = mouse_dir + '_' + sess_dir + '_' + file_name,
            session_id = session_id,
            session_start_time = session_start_time,
            file_create_date = datetime.now(tzlocal()),
            experimenter = experimenter,
            institution = institution,
            related_publications = related_publications
        )
        # -- subject
        subj = pynwb.file.Subject(
            subject_id = mouse_dir,
            age = '',
            description = '',
            genotype = '',
            sex = '',
            species = '',
            weight = ''
        )
        nwbfile.subject = subj
        print(f'NWB file created: {mouse_dir}; {sess_dir}; {session_id}')

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

        # -- Image segmentation processing module
        img_seg_mod = nwbfile.create_processing_module('Image-Segmentation', 'Plane segmentation and ROI identification')

        img_segmentation = nwb_ophys.ImageSegmentation(name = 'img_seg')
        img_seg_mod.add_data_interface(img_segmentation)

        plane_segmentation = nwb_ophys.PlaneSegmentation(
            name = 'pln_seg',
            description = 'description here',
            imaging_plane = imaging_plane
        )
        img_segmentation.add_plane_segmentation([plane_segmentation])

        # add more columns
        plane_segmentation.add_column(name = 'roi_id', description = 'roi id')
        plane_segmentation.add_column(name = 'roi_status', description = 'good or bad ROI')
        plane_segmentation.add_column(name = 'neuron_type', description = 'excitatory or inhibitory')
        plane_segmentation.add_column(name = 'fitness', description = '')
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

        for idx, ival in enumerate(moremat['idx_components']):
            plane_segmentation.add_roi(
                roi_id = ival,
                image_mask = moremat['mask'][:, :, idx],
                roi_status = 'good' if moremat['badROIs01'][idx] == 0 else 'bad',
                fitness = moremat['fitness'][idx],
                neuron_type = 'inhibitory' if neuron_type[idx] == 1 else 'excitatory' if neuron_type[idx] == 0 else 'unknown',
                roi2surr_sig = roi2surr_sig[idx],
                offsets_ch1_pix = offsets_ch1_pix[idx]
            )

        # create a ROI region table
        roi_region = plane_segmentation.create_roi_table_region(
            description = 'good roi region table',
            region = (np.where(moremat['badROIs01'] == 0)[0]).tolist()
        )

        # create another processing module:  trial-segmentation
        trial_seg_mod = nwbfile.create_processing_module('Trial-based',
                                                         'Trial-segmented data based on different event markers')
        dF_F = nwb_ophys.DfOverF(name = 'deconvolved dF-over-F')
        trial_seg_mod.add_data_interface(dF_F)

        # now build "RoiResponseSeries" by ingesting data
        def build_roi_series(data_string_name, post_data, dyn_table):
            data = post_data[data_string_name].traces.transpose([1, 0, 2])
            roi_resp_series = nwb_ophys.RoiResponseSeries(
                name = data_string_name,
                data = data,
                unit = '',
                rois = dyn_table,
                timestamps = post_data[data_string_name].time,
                description = 'ROIs x time x trial'
            )
            return roi_resp_series

        # ingest each trial-based dataset, time-lock to different event types
        data_names = ['firstSideTryAl', 'firstSideTryAl_COM',
                      'goToneAl', 'rewardAl', 'commitIncorrAl',
                      'initToneAl', 'stimAl_allTrs',
                      'stimAl_noEarlyDec', 'stimOffAl']
        for data_name in data_names:
            try:
                dF_F.add_roi_response_series(build_roi_series(data_name, postmat, roi_region))
            except Exception as e:
                print(f'Error adding roi_response_series: {data_name} - ErrorMsg: {str(e)}')

        # -- Write NWB 2.0 file
        save_path = os.path.join('data', 'nwb2.0')
        save_file_name = file_name
        with NWBHDF5IO(os.path.join(save_path, save_file_name), mode = 'w') as io:
            io.write(nwbfile)

        break
    break

