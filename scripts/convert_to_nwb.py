#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from dateutil.tz import tzlocal
import pytz
import re

import json
import numpy as np
import scipy.io as sio
import pynwb
from pynwb import NWBFile, NWBHDF5IO
from pynwb import ophys as nwb_ophys


def main(config_file=sys.argv[1]):
    # ======== Read conversion config file ========
    with open(config_file) as f:
        config = json.load(f)

    # -- Read manifest file, get the list of `.mat` data
    with open(os.path.join(*re.split('/', config['manifest'])), 'r') as f:
        more_mats = [re.search('(?<=\s{2}).*(?=\.mat)', l).group() for l in f if re.search('(more_).*(\.mat)', l)]
        f.seek(0)
        post_mats = [re.search('(?<=\s{2}).*(?=\.mat)', l).group() for l in f if re.search('(post_).*(\.mat)', l)]

    # -- Pairing the 'more_' and 'post_' .mat files to make one dataset for one session
    # so we don't rely on 'more_' and 'post_' .mat being listed in order in the manifest file
    mat_files = []
    for more_mat in more_mats:
        for post_mat in post_mats:
            if re.sub('more_', '', more_mat) == re.sub('post_', '', post_mat):
                mat_files.append((more_mat, post_mat))

    # -- Setup some metadata information
    datetime_format_yymmdd = '%y%m%d'
    timezone = pytz.timezone('US/Eastern')  # assuming the recording is done at NY, NY

    experimenter = config['general']['experimenter']
    institution = config['general']['institution']
    related_publications = config['general']['related_publications']

    # change dir to the data dir
    error_log_file = os.path.abspath(os.path.join(*re.split('/', config['error_log'])))
    save_path = os.path.abspath(os.path.join(*re.split('/', config['output_dir'])))
    os.chdir(os.path.dirname(os.path.abspath(os.path.join(*re.split('/', config['manifest'])))))

    if not os.path.exists(save_path):
        raise NotADirectoryError(f'Invalid output directory: {save_path}')

    # ======== Start conversion ========
    for more_fname, post_fname in mat_files:
        moremat = sio.loadmat(os.path.join(*re.split('/', more_fname)),
                              struct_as_record=False, squeeze_me=True)
        postmat = sio.loadmat(os.path.join(*re.split('/', post_fname)),
                              struct_as_record=False, squeeze_me=True)
        # NWB 2.0
        # step 1: ingest metadata (hard-coded for now ...)
        file_name = re.sub('data/|more_|post_|.mat', '', more_fname)
        mouse_dir = re.split('/', file_name)[0]
        sess_dir = re.split('/', file_name)[1]

        # -- NWB file - a NWB2.0 file for each session
        session_start_time = datetime.strptime(sess_dir, datetime_format_yymmdd)
        session_start_time.astimezone(timezone)
        session_id = ''.join(file_name.split('-')[-2:])  # assuming the last 12-digits is the session id
        nwbfile = NWBFile(
            session_description=file_name,
            identifier=file_name,
            session_id=session_id,
            session_start_time=session_start_time,
            file_create_date=datetime.now(tzlocal()),
            experimenter=experimenter,
            institution=institution,
            related_publications=related_publications)

        # -- subject
        nwbfile.subject = pynwb.file.Subject(
            subject_id=mouse_dir,
            age='',
            description='',
            genotype='',
            sex='',
            species='',
            weight='')

        # -- imaging plane - the plane info ophys was performed on (again hard-coded here)
        device = pynwb.device.Device('img_device')
        nwbfile.add_device(device)
        optical_channel = nwb_ophys.OpticalChannel('Green', 'Green (ET 525/50m)', 525.)  #(nm)
        imaging_plane = nwbfile.create_imaging_plane(
            name='img_pln',
            optical_channel=optical_channel,
            device=device,
            description='imaging plane',
            excitation_lambda=930.,  # (nm)
            imaging_rate=30.,
            indicator='GCaMP6f',
            location='left PPC',
            manifold=np.ones((512, 512, 1)),
            conversion=1e-6,
            unit='micrometer'
        )

        # ------  Epoch ------
        # define custom epoch columns:
        trial_columns = [{'name': 'trial_type', 'description': 'high-rate, low-rate'},
                      {'name': 'trial_pulse_rate', 'description': 'ranged from 5-27Hz, 16Hz boundary for high/low rate'},
                      {'name': 'trial_response', 'description': 'correct, incorrect, no center lick, no go-tone lick'},
                      {'name': 'trial_is_good', 'description': 'good, bad'},
                      {'name': 'init_tone', 'description': '(ms) time of initiation tone w.r.t the start of the trial (t=0)'},
                      {'name': 'stim_onset', 'description': '(ms) time of stimulus onset w.r.t the start of the trial (t=0)'},
                      {'name': 'stim_offset', 'description': '(ms) time of stimulus offset w.r.t the start of the trial (t=0)'},
                      {'name': 'go_tone', 'description': '(ms) time of go tone w.r.t the start of the trial (t=0)'},
                      {'name': 'first_commit', 'description': '(ms) time of first commit w.r.t the start of the trial (t=0)'},
                      {'name': 'second_commit', 'description': '(ms) time of second commit w.r.t the start of the trial (t=0)'}]

        # add new table columns to nwb epoch
        for c in trial_columns:
            nwbfile.add_trial_column(**c)

        # - read and condition data
        outcomes = postmat['outcomes']  # 1: correct, 0: incorrect, nan: no trial, -3: no center lick to start stimulus, -1: no go-tone lick
        outcomes[np.isnan(outcomes)] = -10  # replace nan with -10 to easily make outcome dict
        trial_response_dict = {1: 'correct',
                               0: 'incorrect',
                               -1: 'no go-tone lick (-1)',  # because from the data, it seems like code(-1) and code(-4) both refer to 'no go-tone lick'
                               -4: 'no go-tone lick (-4)',
                               -3: 'no center lick',
                               -2: 'no first commit',
                               -5: 'no second commit',
                               -10: 'no trial'}

        # get timeInitTone and handle some timeInitTone elements being vectors instead of scalars (get [0] of that vector)
        init_tone = [a if np.isscalar(a) else a[0] for a in postmat['timeInitTone']]
        # merge timeReward and timeCommitIncorrectResp to get an overall second commit times
        timeCommitIncorrResp = postmat['timeCommitIncorrResp']
        second_commit_times = postmat['timeReward']
        ix = ~np.isnan(timeCommitIncorrResp)
        second_commit_times[ix] = timeCommitIncorrResp[ix]

        # get trial start stop times
        if 'alldata_frameTimes' not in postmat:
            start_time = np.full(outcomes.shape, np.nan)
            stop_time = np.full(outcomes.shape, np.nan)
        else:
            alldata_frameTimes = postmat['alldata_frameTimes']  # timestamps of each trial for all trials
            start_time = [t[0] for t in alldata_frameTimes]
            stop_time = [t[-1] for t in alldata_frameTimes]

        # - now insert each trial into trial table
        for k in range(outcomes.size):
            nwbfile.add_trial(start_time=start_time[k],
                              stop_time=stop_time[k],
                              trial_type=('High-rate' if postmat['stimrate'][k] >= 16 else 'Low-rate'),
                              trial_pulse_rate=postmat['stimrate'][k],
                              trial_response=trial_response_dict[outcomes[k]],
                              trial_is_good=(outcomes[k] >= 0),
                              init_tone=init_tone[k],
                              stim_onset=postmat['timeStimOnsetAll'][k],
                              stim_offset=postmat['timeSingleStimOffset'][k],
                              go_tone=postmat['timeCommitCL_CR_Gotone'][k],
                              first_commit=postmat['time1stSideTry'][k],
                              second_commit=second_commit_times[k])

        # ------ Image Segmentation processing module ------
        img_seg_mod = nwbfile.create_processing_module('Image-Segmentation',
                                                       'Plane segmentation and ROI identification')

        img_segmentation = nwb_ophys.ImageSegmentation(name='img_seg')
        img_seg_mod.add_data_interface(img_segmentation)

        plane_segmentation = nwb_ophys.PlaneSegmentation(
            name='pln_seg',
            description='description here',
            imaging_plane=imaging_plane)
        img_segmentation.add_plane_segmentation([plane_segmentation])

        # add more columns
        plane_segmentation.add_column(name='roi_id', description='roi id')
        plane_segmentation.add_column(name='roi_status', description='good or bad ROI')
        plane_segmentation.add_column(name='neuron_type', description='excitatory or inhibitory')
        plane_segmentation.add_column(name='fitness', description='')
        plane_segmentation.add_column(name='roi2surr_sig', description='')
        plane_segmentation.add_column(name='offsets_ch1_pix', description='')

        # start inserting ROI mask
        bad_roi_mask = np.where(moremat['badROIs01'] == 0)
        neuron_type = np.full_like(moremat['idx_components'], np.nan)
        neuron_type[bad_roi_mask] = moremat['inhibitRois_pix']
        roi2surr_sig = np.full_like(moremat['idx_components'], np.nan)
        roi2surr_sig[bad_roi_mask] = moremat['roi2surr_sig']
        offsets_ch1_pix = np.full_like(moremat['idx_components'], np.nan)
        offsets_ch1_pix[bad_roi_mask] = moremat['offsets_ch1_pix']

        neuron_type_dict = {0: 'excitatory', 1: 'inhibitory'}
        neuron_status_dict = {0: 'good', 1: 'bad'}
        for idx in range(moremat['idx_components'].size):
            plane_segmentation.add_roi(
                roi_id=moremat['idx_components'][idx],
                image_mask=moremat['mask'][:, :, idx],
                roi_status=neuron_status_dict.get(moremat['badROIs01'][idx]),
                fitness=moremat['fitness'][idx],
                neuron_type=neuron_type_dict.get(neuron_type[idx], 'unknown'),
                roi2surr_sig=roi2surr_sig[idx],
                offsets_ch1_pix=offsets_ch1_pix[idx])

        # create a ROI region table
        roi_region = plane_segmentation.create_roi_table_region(
            description='good roi region table',
            region=(np.where(moremat['badROIs01'] == 0)[0]).tolist())

        # ------ Trial Segmentation processing module ------
        trial_seg_mod = nwbfile.create_processing_module('Trial-based',
                                                         'Trial-segmented data based on different event markers')
        dF_F = nwb_ophys.DfOverF(name='deconvolved dF-over-F')
        trial_seg_mod.add_data_interface(dF_F)

        # now build "RoiResponseSeries" by ingesting data
        def build_roi_series(data_string_name, post_data, dyn_table):
            return nwb_ophys.RoiResponseSeries(
                name=data_string_name,
                data=post_data[data_string_name].traces.transpose([1, 0, 2]),
                unit='',
                rois=dyn_table,
                timestamps=post_data[data_string_name].time,
                description=f'(ROIs x time x trial), aligned to event_id: {post_data[data_string_name].eventI}')


        # ingest each trial-based dataset, time-lock to different event types
        data_names = ['firstSideTryAl',
                      'firstSideTryAl_COM',
                      'goToneAl',
                      'rewardAl',
                      'commitIncorrAl',
                      'initToneAl',
                      'stimAl_allTrs',
                      'stimAl_noEarlyDec',
                      'stimOffAl']
        for data_name in data_names:
            try:
                dF_F.add_roi_response_series(build_roi_series(data_name, postmat, roi_region))
            except Exception as e:
                with open(error_log_file, 'a+') as error_log:
                    error_log.write(f'Error adding roi_response_series: {data_name}\n\t\tErrorMsg: {str(e)}\n')

        # ------ Behavior processing module ------
        behavior_mod = nwbfile.create_processing_module('Behavior',
                                                         'Behavior data (e.g. wheel revolution, lick traces)')
        behav_epoch = pynwb.behavior.BehavioralTimeSeries(name='Epoched behavioral data, time-locked to experimental event')
        behavior_mod.add_data_interface(behav_epoch)

        behavioral_names = ['firstSideTryAl_wheelRev', 'firstSideTryAl_lick']
        for behav in behavioral_names:
            behav_epoch.create_timeseries(
                name=behav,
                data=postmat[behav].traces,
                timestamps=postmat[behav].time,
                description=f'(time x trial), aligned to event_id: {postmat[behav].eventI}')

        # ------ Write NWB 2.0 file ------
        save_file_name = ''.join([re.sub('/', '_', file_name), '.nwb'])
        with NWBHDF5IO(os.path.join(save_path, save_file_name), mode='w') as io:
            io.write(nwbfile)
            print(f'Write NWB 2.0 file: {save_file_name}')


if __name__ == "__main__":
    sys.exit(main())