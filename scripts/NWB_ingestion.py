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
        session_id = file_name.split('-')[-2] + file_name.split('-')[-1]  # assuming the last 12-digits is the session id
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
        device = pynwb.device.Device('img_device')
        nwbfile.add_device(device)
        optical_channel = nwb_ophys.OpticalChannel('Red', 'Red (ET670/50m)', 670.)
        imaging_plane = nwbfile.create_imaging_plane(
            name = 'img_pln',
            optical_channel = optical_channel,
            device = device,
            description = 'imaging plane',
            excitation_lambda = 930.,
            imaging_rate = 30.,
            indicator = 'tdTomato',
            location = 'left PPC',
            manifold = np.ones((512, 512, 1)),
            conversion = 1e-6,
            unit = 'micrometer'
        )

        # -- Epoch
        # define epoch table
        nwbfile.add_trial_column(name = 'trial_type', description = 'high-rate, low-rate')
        nwbfile.add_trial_column(name = 'trial_pulse_rate', description = 'ranged from 5-27Hz, 16Hz boundary for high/low rate')
        nwbfile.add_trial_column(name = 'trial_response', description = 'correct, incorrect, no center lick, no go-tone lick')
        nwbfile.add_trial_column(name = 'trial_is_good', description = 'good, bad')

        nwbfile.add_trial_column(name = 'init_tone', description = '(in ms) time of initiation tone w.r.t the start of the trial (t=0)')
        nwbfile.add_trial_column(name = 'stim_onset', description = '(in ms) time of stimulus onset w.r.t the start of the trial (t=0)')
        nwbfile.add_trial_column(name = 'stim_offset', description = '(in ms) time of stimulus offset w.r.t the start of the trial (t=0)')
        nwbfile.add_trial_column(name = 'go_tone', description = '(in ms) time of go tone w.r.t the start of the trial (t=0)')
        nwbfile.add_trial_column(name = 'first_commit', description = '(in ms) time of first commit w.r.t the start of the trial (t=0)')
        nwbfile.add_trial_column(name = 'second_commit', description = '(in ms) time of second commit w.r.t the start of the trial (t=0)')

        # - read and condition data
        outcomes = postmat['outcomes']  # 1: correct, 0: incorrect, nan: no trial, -3: no center lick to start stimulus, -1: no go-tone lick
        outcomes[np.isnan(outcomes)] = -10  # replace nan with -10 to easily make outcome dict
        trial_response_dict = {1: 'correct', 0: 'incorrect', -1: 'no go-tone lick (-1)',
                               -4: 'no go-tone lick (-4)', -3: 'no center lick', -2: 'no first commit',
                               -5: 'no second commit', -10: 'no trial'}
        stimrate = postmat['stimrate']  # stim rate

        timeInitTone = postmat['timeInitTone']  # init tone

        # handling some timeInitTone elements being vectors instead of scalars (get [0] of that vector)
        def get_0th(a):
            return a if np.isscalar(a) else a[0]

        init_tone = [get_0th(a) for a in timeInitTone]  # init tone

        timeStimOnsetAll = postmat['timeStimOnsetAll']  # stim onset
        timeSingleStimOffset = postmat['timeSingleStimOffset']  # stim offset
        timeCommitCL_CR_Gotone = postmat['timeCommitCL_CR_Gotone']  # go tone
        time1stSideTry = postmat['time1stSideTry']  # correct and incorrect 1st commit

        timeReward = postmat['timeReward']
        timeCommitIncorrResp = postmat['timeCommitIncorrResp']
        # merge timeReward and timeCommitIncorrectResp to get an overall second commit times
        second_commit_times = timeReward.copy()
        second_commit_times[~np.isnan(timeCommitIncorrResp)] = timeCommitIncorrResp[~np.isnan(timeCommitIncorrResp)]

        alldata_frameTimes = postmat['alldata_frameTimes']  # timestamps of each trial for all trials

        # get trial start stop times
        alldata_frameTimes = postmat['alldata_frameTimes']  # timestamps of each trial for all trials
        start_time = [t[0] for t in alldata_frameTimes]
        stop_time = [t[-1] for t in alldata_frameTimes]

        # check good trial         trial_is_good = [(a == 1 or a == 0) for a in outcomes]
        tags = [(a == 1 or a == 0) for a in outcomes] # make some random tags for testing TODO rmv

        # - now insert each trial into trial table
        for k in np.arange(outcomes.size):
            nwbfile.add_trial(start_time=start_time[k],
                                     stop_time=stop_time[k],
                                     trial_type=('High-rate' if stimrate[k] >= 16 else 'Low-rate'),
                                     trial_pulse_rate=stimrate[k],
                                     trial_response=trial_response_dict[outcomes[k]],
                                     trial_is_good=(outcomes[k] >= 0),
                                     init_tone=init_tone[k],
                                     stim_onset=timeStimOnsetAll[k],
                                     stim_offset=timeSingleStimOffset[k],
                                     go_tone=timeCommitCL_CR_Gotone[k],
                                     first_commit=time1stSideTry[k],
                                     second_commit=second_commit_times[k])

        # -- Image segmentation processing module
        img_seg_mod = nwbfile.create_processing_module('Image-Segmentation',
                                                       'Plane segmentation and ROI identification')

        img_segmentation = nwb_ophys.ImageSegmentation(name='img_seg')
        img_seg_mod.add_data_interface(img_segmentation)

        plane_segmentation = nwb_ophys.PlaneSegmentation(
            name='pln_seg',
            description='description here',
            imaging_plane=imaging_plane
        )
        img_segmentation.add_plane_segmentation([plane_segmentation])

        # add more columns
        plane_segmentation.add_column(name='roi_id', description='roi id')
        plane_segmentation.add_column(name='roi_status', description='good or bad ROI')
        plane_segmentation.add_column(name='neuron_type', description='excitatory or inhibitory')
        plane_segmentation.add_column(name='fitness', description='')
        plane_segmentation.add_column(name='roi2surr_sig', description='')
        plane_segmentation.add_column(name='offsets_ch1_pix', description='')

        # start inserting ROI mask
        tmp = np.empty(moremat['idx_components'].shape)
        tmp.fill(np.nan)
        neuron_type = tmp.copy()
        neuron_type[np.where(moremat['badROIs01'] == 0)] = moremat['inhibitRois_pix']
        roi2surr_sig = tmp.copy()
        roi2surr_sig[np.where(moremat['badROIs01'] == 0)] = moremat['roi2surr_sig']
        offsets_ch1_pix = tmp.copy()
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
        save_file_name = f'{mouse_dir}_{file_name}.nwb'
        with NWBHDF5IO(os.path.join(save_path, save_file_name), mode = 'w') as io:
            io.write(nwbfile)
            print(f'Write NWB 2.0 file: {save_file_name}')

