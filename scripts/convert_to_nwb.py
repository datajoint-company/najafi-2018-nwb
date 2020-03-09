#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from dateutil.tz import tzlocal
from pathlib import Path
import pytz
import re
import json
import numpy as np
import scipy.io as sio
import pynwb
from pynwb import NWBFile, NWBHDF5IO, ophys as nwb_ophys
from collections import defaultdict
import tqdm

# Read configuration
try:
    config_file = sys.argv[1]
except IndexError:
    config_file = 'conversion_config.json'
with open(config_file) as f:
    config = json.load(f)

# Read the list of .mat files from manifest file
file_pattern = re.compile(
    r'(?P<checksum>[0-9a-f]{32})  (?P<path>.+)(?P<prefix>(post|more))_(?P<session>.+)(?P<ext>\.mat)')
mat_file_pairs = defaultdict(dict)
with open(Path(config['manifest']), 'r') as f:
    for line in f:
        match = file_pattern.match(line)
        if match:
            mat_file_pairs[match['session']][match['prefix']] = Path('{root}/{path}/{prefix}_{session}{ext}'.format(
                root=os.path.dirname(os.path.abspath(Path(config['manifest']))), **match.groupdict()))

# save an NWB file for each session
save_path = os.path.abspath(Path(config['output_dir']))
mat_file_pairs_=[(list(mat_file_pairs.keys())[0],list(mat_file_pairs.values())[0])]
for session, file_pair in tqdm.tqdm(mat_file_pairs_):
    moremat, postmat = (sio.loadmat(file_pair[x], struct_as_record=False, squeeze_me=True)
                        for x in ('more', 'post'))
    mouse_folder, session_folder = file_pair['more'].parts[-3:-1]

    nwbfile = NWBFile(
        session_description=session,
        identifier=session,
        session_id=''.join(session.split('-')[-2:]),  # use the last 12-digits as the session id
        session_start_time=datetime.strptime(session_folder, '%y%m%d').astimezone(pytz.timezone('US/Eastern')),
        file_create_date=datetime.now(tzlocal()),
        **config['general'])

    nwbfile.subject = pynwb.file.Subject(
        subject_id=mouse_folder, age='', description='', genotype='', sex='', species='', weight='')

    # -- imaging plane - the plane info ophys was performed on (again hard-coded here)
    device = pynwb.device.Device('img_device')
    nwbfile.add_device(device)
    imaging_plane = nwbfile.create_imaging_plane(
        name='ImagingPlane',
        optical_channel=nwb_ophys.OpticalChannel('Green', 'Green (ET 525/50m)', 525.),  #(nm)
        device=device,
        description='imaging plane',
        excitation_lambda=930.,  # (nm)
        imaging_rate=30.,
        indicator='GCaMP6f',
        location='left PPC',
        conversion=1e-6,
        unit='micrometer')

    # Epochs: define custom epoch columns
    trial_columns = dict(
        trial_type='high-rate, low-rate',
        trial_pulse_rate='ranged from 5-27Hz, 16Hz boundary for high/low rate',
        trial_response='correct, incorrect, no center lick, no go-tone lick',
        trial_is_good='good, bad',
        init_tone='(sec) time of initiation tone w.r.t the start of the trial (t=0)',
        stim_onset='(sec) time of stimulus onset w.r.t the start of the trial (t=0)',
        stim_offset='(sec) time of stimulus offset w.r.t the start of the trial (t=0)',
        go_tone='(sec) time of go tone w.r.t the start of the trial (t=0)',
        first_commit='(sec) time of first commit w.r.t the start of the trial (t=0)',
        second_commit='(sec) time of second commit w.r.t the start of the trial (t=0)'
    )
    for k, v in trial_columns.items():
        nwbfile.add_trial_column(name=k, description=v)

    # - read and condition data
    outcomes = postmat['outcomes']  # 1: correct, 0: incorrect, nan: no trial, -3: no center lick to start stimulus, -1: no go-tone lick
    outcomes[np.isnan(outcomes)] = -10  # replace nan with -10 to easily make outcome dict
    trial_response_dict = {
        1: 'correct',
        0: 'incorrect',
        -1: 'no go-tone lick (-1)',  # because from the data, it seems like code(-1) and code(-4) both refer to 'no go-tone lick'
        -4: 'no go-tone lick (-4)',
        -3: 'no center lick',
        -2: 'no first commit',
        -5: 'no second commit',
        -10: 'no decision'}

    # get timeInitTone and handle some timeInitTone elements being vectors instead of scalars (get [0] of that vector)
    init_tone = [a if np.isscalar(a) else a[0] for a in postmat['timeInitTone']]
    # merge timeReward and timeCommitIncorrectResp to get an overall second commit times
    second_commit_times = postmat['timeReward']
    ix = ~np.isnan(postmat['timeCommitIncorrResp'])
    second_commit_times[ix] = postmat['timeCommitIncorrResp'][ix]

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
        nwbfile.add_trial(
            start_time=start_time[k]/1000,
            stop_time=stop_time[k]/1000,
            trial_type=('High-rate' if postmat['stimrate'][k] >= 16 else 'Low-rate'),
            trial_pulse_rate=postmat['stimrate'][k],
            trial_response=trial_response_dict[outcomes[k]],
            trial_is_good=(outcomes[k] >= 0),
            init_tone=init_tone[k]/1000,#in seconds
            stim_onset=postmat['timeStimOnsetAll'][k]/1000,
            stim_offset=postmat['timeSingleStimOffset'][k]/1000,
            go_tone=postmat['timeCommitCL_CR_Gotone'][k]/1000,
            first_commit=postmat['time1stSideTry'][k]/1000,
            second_commit=second_commit_times[k]/1000)

    # ------ Image Segmentation processing module ------
    img_seg_mod = nwbfile.create_processing_module(
        'Ophys', 'Plane segmentation and ROI information')
    img_segmentation = nwb_ophys.ImageSegmentation(name='ImageSegmentation')
    img_seg_mod.add_data_interface(img_segmentation)
    plane_segmentation = nwb_ophys.PlaneSegmentation(
        name='PlaneSegmentation',
        description='description here',
        imaging_plane=imaging_plane)
    img_segmentation.add_plane_segmentation([plane_segmentation])

    # add segmentation columns
    for k, v in dict(
            roi_id='roi id',
            roi_status='good or bad ROI',
            neuron_type='excitatory or inhibitory',
            fitness='',
            roi2surr_sig='',
            offsets_ch1_pix='').items():
        plane_segmentation.add_column(name=k, description=v)

    # insert ROI mask
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

    # ingest each trial-based dataset, time-lock to different event types
    for data_name in ('firstSideTryAl', 'firstSideTryAl_COM', 'goToneAl', 'rewardAl', 'commitIncorrAl',
                      'initToneAl', 'stimAl_allTrs', 'stimAl_noEarlyDec', 'stimOffAl'):
        try:
            dF_F = nwb_ophys.DfOverF(name = f'dFoF_{data_name}')
            for tr_idx, d in enumerate(postmat[data_name].traces.transpose([2, 1, 0])):
                dF_F.add_roi_response_series(
                    nwb_ophys.RoiResponseSeries(
                        name=f'Trial_{tr_idx:02d}',
                        data=d.T,
                        unit='',
                        rois=roi_region,
                        timestamps=postmat[data_name].time/1000,#in seconds
                        description=f'(ROIs x time), aligned to event_id: {postmat[data_name].eventI}'))
            img_seg_mod.add_data_interface(dF_F)
        except Exception as e:
            print(f'Error adding roi_response_series: {data_name}\n\t\tErrorMsg: {str(e)}\n', file=sys.stderr)

    # ------ Behavior processing module ------
    behavior_mod = nwbfile.create_processing_module(
        'Behavior', 'Behavior data (e.g. wheel revolution, lick traces)')
    behavior_epoch = pynwb.behavior.BehavioralTimeSeries(
        name='Epoched_behavioral_series')
    behavior_mod.add_data_interface(behavior_epoch)

    for behavior in ['firstSideTryAl_wheelRev', 'firstSideTryAl_lick']:
        behavior_epoch.create_timeseries(
            name=behavior,
            data=postmat[behavior].traces,
            timestamps=postmat[behavior].time/1000,#in seconds
            description=f'(time x trial), aligned to event_id: {postmat[behavior].eventI}')

    with NWBHDF5IO(os.path.join(save_path, mouse_folder + '_' + session + '.nwb'), mode='w') as io:
        io.write(nwbfile)

