# Code adapted from https://github.com/denmanlab/dlab/blob/master/nwbtools.py
import glob
from typing import Optional, Union
import pandas as pd
import numpy as np
import os
import csv
from scipy.io import loadmat


_option234_positions = np.zeros((384, 2))
_option234_positions[:, 0][::4] = 21
_option234_positions[:, 0][1::4] = 53
_option234_positions[:, 0][2::4] = 5
_option234_positions[:, 0][3::4] = 37
_option234_positions[:, 1] = np.floor(np.linspace(383, 0, 384) / 2) * 20

OPTION234_POSITIONS = _option234_positions


def get_peak_waveform_from_template(template: np.ndarray) -> np.ndarray:
    """
    Get the peak waveform from a given template

    :param template: Template from an NWB
    :return: Numpy array of the waveform
    """

    tmpl_max = 0
    ind = 0
    peak = np.zeros(np.shape(template.T)[0])
    for i, wv in enumerate(template.T):
        if np.max(np.abs(wv)) > tmpl_max:
            tmpl_max = np.max(np.abs(wv))
            ind = i
            peak = wv
    return peak


def load_phy_template(path: str, site_positions: Optional[np.ndarray] = None, samplingrate: float = 30000.) -> dict:
    """
    Load spike data that has been manually sorted with the phy-template GUI
    the site_positions should contain coordinates of the channels in probe space. for example, in um on the face of
    the probe

    :param path: Path to the Phy data
    :param site_positions: Positions of the channels on the probe
    :param samplingrate: Rate at which the Neuropixels probe sampled the data, defaults to 30kHz

    :return: A dictionary of 'good' units with keys
        times: spike times, in seconds
        template: template used for matching
        ypos: y position on the probe, calculated from the template. requires an accurate site_positions. averages template from 100 spikes.
        xpos: x position on the probe, calcualted from the template. requires an accurate site_positions. averages template from 100 spikes.

    """
    if site_positions is None:
        site_positions = OPTION234_POSITIONS

    fps = []  # List of file pointers ("file-like" objects) to close

    def open_fp(filename):
        myfp = open(os.path.join(path, filename), 'rb')
        fps.append(myfp)
        return myfp
    try:
        clusters = np.load(open_fp('spike_clusters.npy'))
        spikes = np.load(open_fp('spike_times.npy'))
        spike_templates = np.load(open_fp('spike_templates.npy'))
        templates = np.load(open_fp('templates.npy'))

        cluster_id, KSlabel, KSamplitude, KScontamination = [], [], [], []
        [KSlabel.append(row) for row in csv.reader(open_fp('cluster_KSLabel.tsv'))]
        [KSamplitude.append(row) for row in csv.reader(open_fp('cluster_Amplitude.tsv'))]
        [KScontamination.append(row) for row in csv.reader(open_fp('cluster_ContamPct.tsv'))]
        if os.path.isfile(os.path.join(path, 'cluster_group.tsv')):
            # cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_group.tsv')))][1:]
            [cluster_id.append(row) for row in csv.reader(open_fp('cluster_group.tsv'))]
        else:
            if os.path.isfile(os.path.join(path, 'cluster_groups.csv')):
                # cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))][1:]
                [cluster_id.append(row) for row in csv.reader(open_fp('cluster_groups.csv'))]
            else:
                print('cant find cluster groups, either .tsv or .csv')

        units = {}
        for i in np.arange(1, np.shape(cluster_id)[0]):
            unit = int(cluster_id[i][0].split('\t')[0])
            units[str(unit)] = {}

            # get the unit spike times
            units[str(unit)]['samples'] = spikes[np.where(clusters == unit)].flatten()
            units[str(unit)]['times'] = spikes[np.where(clusters == unit)] / samplingrate
            units[str(unit)]['times'] = units[str(unit)]['times'].flatten()

            # get the mean template used for this unit
            all_templates = spike_templates[np.where(clusters == unit)].flatten()
            n_templates_to_subsample = 100
            random_subsample_of_templates = templates[
                all_templates[np.array(np.random.rand(n_templates_to_subsample) * all_templates.shape[0]).astype(int)]]
            mean_template = np.mean(random_subsample_of_templates, axis=0)
            units[str(unit)]['template'] = mean_template

            # take a weighted average of the site_positions, where the weights is the absolute value of the template for that channel
            # this gets us the x and y positions of the unit on the probe.
            # print(mean_template.T.shape)
            weights = np.zeros(site_positions.shape)
            for channel in range(mean_template.T.shape[0]):
                weights[channel, :] = np.trapz(np.abs(mean_template.T[channel, :]))
            weights = weights / np.max(weights)
            low_values_indices = weights < 0.25  # Where values are low,
            weights[low_values_indices] = 0  # make the weight 0
            (xpos, ypos) = np.average(site_positions, axis=0, weights=weights)
            units[str(unit)]['waveform_weights'] = weights
            units[str(unit)]['xpos'] = xpos
            units[str(unit)]['ypos'] = ypos  # - site_positions[-1][1]
            units[str(unit)]['label'] = cluster_id[i][0].split('\t')[1]
            units[str(unit)]['KSlabel'] = KSlabel[i][0].split('\t')[1]
            units[str(unit)]['KSamplitude'] = KSamplitude[i][0].split('\t')[1]
            units[str(unit)]['KScontamination'] = KScontamination[i][0].split('\t')[1]

        return units
    finally:
        for fp in fps:  # Close all file pointers
            fp.close()


def df_from_phy(path: str, site_positions: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Create a pandas dataframe from phy data

    :param path: Path to the Phy data
    :param site_positions: Positions of the channels on the probe

    :return: Pandas Dataframe
    """
    if site_positions is None:
        site_positions = OPTION234_POSITIONS

    nwb_data = load_phy_template(path, site_positions)
    # structures is a dictionary that defines the bounds of the structure e.g.:{'v1':(0,850), 'hpc':(850,2000)}
    mouse = []
    experiment = []
    cell = []
    ypos = []
    xpos = []
    waveform = []
    template = []
    # structure = []  #  Missing in original code, ignoring
    times = []
    index = []
    count = 1
    # nwb_id = []  # Missin in original code, ignoring
    probe_id = []
    depth = []  # print(list(nwb_data.keys()));print(list(nwb_data['processing'].keys()));
    for probe in list(nwb_data['processing'].keys()):
        if 'UnitTimes' in list(nwb_data['processing'][probe].keys()):
            for i, u in enumerate(list(nwb_data['processing'][probe]['UnitTimes'].keys())):
                if u != 'unit_list':
                    # nwb_id.append(nwbid)  Missing from original code, ignoring
                    probe_id.append(probe)
                    index.append(count)
                    count += 1
                    mouse.append(str(np.array(nwb_data.get('identifier'))))
                    experiment.append(1)
                    cell.append(u)
                    times.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['times']))
                    # print(list(nwb_data['processing'][probe]['UnitTimes'][u].keys()))
                    if 'ypos' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                        ypos.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['ypos']))
                        has_ypos = True
                    else:
                        ypos.append(None)
                        has_ypos = False
                    if 'depth' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                        depth.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['depth']))
                    else:
                        if has_ypos:
                            depth.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['ypos']))
                        else:
                            depth.append(None)
                    if 'xpos' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                        xpos.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['xpos']))
                        has_xpos = True
                    else:
                        xpos.append(None)
                        has_xpos = False
                    template.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['template']))
                    waveform.append(get_peak_waveform_from_template(template[-1]))
                    # if not structures == None:  # Missing in original code, ignoring
                    #     structur = None
                    #     for struct, bounds in structures.iteritems():
                    #         if ypos[-1] > bounds[0] and ypos[-1] < bounds[1]:
                    #             structur = struct
                    # else:
                    #     structur = None
                    # structure.append(structur)
    df = pd.DataFrame(index=index)
    df = df.fillna(np.nan)
    # df['nwb_id'] = nwb_id  # Missing from original code, ignoring
    df['mouse'] = mouse
    df['experiment'] = experiment
    df['probe'] = probe_id
    # df['structure'] = structure  # Missin in original code, ignoring
    df['cell'] = cell
    df['times'] = times
    df['ypos'] = ypos
    df['xpos'] = xpos
    df['depth'] = depth
    df['waveform'] = waveform
    df['template'] = template
    return df


def load_unit_data(recording_path: str, probe_depth: int = 3840, site_positions: Optional[np.ndarray] = None,
                   probe_name: Optional[str] = None, spikes_filename: str = 'spike_secs.npy', aligned: bool = True,
                   df: bool = True, **kwargs) -> Union[pd.DataFrame, list[dict]]:
    """
    Load the unit data from a given recording

    :param recording_path: path to the recording str
    :param probe_depth: depth of the probe
    :param site_positions: Channel positions on the neuropixels probe
    :param probe_name: Name of the probe, defaults to the recording path
    :param spikes_filename: Spikes filename, defaults to spike_secs.npy
    :param aligned: bool if the spike times are aligned along the sampling rate, if not will be converted
    :param df: Return data as a dataframe

    :return: Dict or Dataframe of the unit data
    """

    if site_positions is None:
        site_positions = OPTION234_POSITIONS

    if probe_name is None:
        probe_name = recording_path
    # Get individual folders for each probe
    unit_times = []
    if not aligned:
        # if not sampling_rate:
        #     imec_meta = readAPMeta(recording_path+'\\') #extract meta file
        #     sampRate = float(imec_meta['imSampRate']) #get sampling rate (Hz)
        # else:
        if 'sampling_rate' in kwargs.keys():
            sampRate = float(kwargs['sampling_rate'])
        else:
            sampRate = 30000
        spike_times = np.ndarray.flatten(np.load(os.path.join(recording_path, 'spike_times.npy')))/sampRate
    else:
        spike_times = np.ndarray.flatten(np.load(os.path.join(recording_path, spikes_filename)))

    cluster_info = pd.read_csv(os.path.join(recording_path, 'cluster_info.tsv'), '\t')
    if cluster_info.keys()[0] == 'cluster_id':
        cluster_info = cluster_info.rename(columns={'cluster_id': 'id'})
    spike_clusters = np.ndarray.flatten(np.load(os.path.join(recording_path, 'spike_clusters.npy')))
    spike_templates = np.load(open(os.path.join(recording_path,'spike_templates.npy'),'rb'))
    templates = np.load(open(os.path.join(recording_path,'templates.npy'),'rb'))
    amplitudes = np.load(open(os.path.join(recording_path,'amplitudes.npy'),'rb'))
    weights = np.zeros(site_positions.shape)

    # Generate Unit Times Table
    for index, unitID in enumerate(cluster_info['id'].values):
        # get mean template used for each unit
        all_templates = spike_templates[np.where(spike_clusters==unitID)].flatten()
        n_templates_to_subsample = 100
        random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
        mean_template = np.mean(random_subsample_of_templates,axis=0)

        # take a weighted average of the site_positions, where the weights is the absolute value of the template for
        # that channel
        # this gets us the x and y positions of the unit on the probe.
        for channel in range(mean_template.T.shape[0]):
            weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
        weights = weights/np.max(weights)
        low_values_indices = weights < 0.25  # Where values are low,
        weights[low_values_indices] = 0      # make the weight 0
        (xpos,zpos)=np.average(site_positions, axis=0, weights=weights)

        unit_times.append({'probe':probe_name,
                           'unit_id': unitID,
                           'group': cluster_info.group[index],
                           # 'depth':cluster_info.depth[index],
                           'depth': (zpos-3840)+probe_depth,
                           'xpos': xpos,
                           'zpos': zpos,
                           'no_spikes': cluster_info.n_spikes[index],
                           'KSlabel': cluster_info['KSLabel'][index],
                           'KSamplitude':cluster_info.Amplitude[index],
                           'KScontamination': cluster_info.ContamPct[index],
                           'template': mean_template,
                           'waveform_weights': weights,
                           'amplitudes': amplitudes[:,0][spike_clusters==unitID],
                           'times': spike_times[spike_clusters == unitID],
                            })
    if df:
        unit_data = pd.DataFrame(unit_times)
        # Remove clusters with no associated spike times left over from Phy
        for i,j in enumerate(unit_data.times):
            if len(unit_data.times[i])==0:
                unit_data.times[i]='empty'
        unit_times = unit_data[unit_data.times!='empty']
        return (unit_times)
    else:
        return (unit_times)


def load_unit_data_from_phy(recording_path: str, chanmap: Optional[dict] = None, insertion_depth: int = 3840,
                            insertion_angle: float = 0) -> dict:
    """
    Load unit data from Phy
    Requires that phy has been run to generate cluster_info.tsv
    searches the folder for the chanmap the KS used, or searches one folder up for it

    :param recording_path: Path to the recording data
    :param chanmap: Channel map, if not provided will attempt to find a file matching '\*hanMap.mat'
    :param insertion_angle: insertion angle
    :param insertion_depth: insertion depth

    :return: dict of unit data
    """

    cluster_info = pd.read_csv(os.path.join(recording_path, 'cluster_info.tsv'), '\t')
    if cluster_info.keys()[0] == 'cluster_id':
        cluster_info = cluster_info.rename(columns={'cluster_id': 'id'})
    spike_clusters = np.ndarray.flatten(np.load(os.path.join(recording_path, 'spike_clusters.npy')))
    spike_templates = np.load(open(os.path.join(recording_path, 'spike_templates.npy'), 'rb'))
    templates = np.load(open(os.path.join(recording_path, 'templates.npy'), 'rb'))
    spike_times = np.load(open(os.path.join(recording_path, 'spike_times.npy'), 'rb'))
    timestamps = np.load(open(os.path.join(recording_path, 'timestamps.npy'), 'rb'))
    spike_secs = timestamps[spike_times.flatten()]

    # Parse spike times for each unit. also get the template so we can use it for waveform shape clustering
    times = []
    mean_templates = []
    for unitID in cluster_info.id.values:
        times.append(spike_secs[spike_clusters == unitID])

        all_templates = spike_templates[np.where(spike_clusters == unitID)].flatten()
        if len(all_templates) > 100:
            n_templates_to_subsample = 100
        else:
            n_templates_to_subsample = len(all_templates)
        random_subsample_of_templates = templates[
            all_templates[np.array(np.random.rand(n_templates_to_subsample) * all_templates.shape[0]).astype(int)]]
        mean_template = np.mean(random_subsample_of_templates, axis=0)
        mean_templates.append(mean_template)
    cluster_info['times'] = times
    cluster_info['template'] = mean_templates
    cluster_info['depth_from_pia'] = cluster_info.depth.values * -1 + insertion_depth * np.cos(
        np.deg2rad(insertion_angle))

    if chanmap is None:
        try:
            chanmap = loadmat(glob.glob(os.path.join(recording_path, '*hanMap.mat'))[0])
        except:
            chanmap = loadmat(glob.glob(os.path.join(os.path.dirname(recording_path), '*hanMap.mat'))[0])

    cluster_info['ycoords'] = chanmap['ycoords'].flatten()[cluster_info.ch.values]
    cluster_info['xcoords'] = chanmap['xcoords'].flatten()[cluster_info.ch.values]
    cluster_info['shank'] = np.floor(cluster_info['xcoords'].values / 205.).astype(int)

    return cluster_info


def recreate_probe_timestamps_from_TTL(directory: str) -> None:
    """
    Recreate probe timestamps from a TTL feed. Will create files new_timestamps/'templates.npy' and
    new_timestamps/'sample_numbers.npy'

    :param directory: str to the directory
    :returns: None, will create files on FS
    """

    probe = directory.split('-AP')[0][-1]
    recording_base = os.path.dirname(os.path.dirname(directory))

    with open(os.path.join(recording_base, 'sync_messages.txt')) as f:
        lines = f.readlines()
        for line in lines:
            if 'Probe' + probe + '-AP' in line:
                cont_start_sample = int(line.split(':')[1][1:].split('\n')[0])
    f.close()

    TTL_samples = np.load(
        os.path.join(glob.glob(os.path.join(recording_base, 'events') + '/*Probe' + probe + '*AP*')[0], 'TTL',
                     'sample_numbers.npy'))[::2]
    TTL_timestamps = np.load(
        os.path.join(glob.glob(os.path.join(recording_base, 'events') + '/*Probe' + probe + '*AP*')[0], 'TTL',
                     'timestamps.npy'))[::2]

    cont_raw = np.memmap(os.path.join(directory, 'continuous.dat'), dtype=np.int16)
    cont_samples = np.arange(cont_start_sample, cont_start_sample + (int(cont_raw.shape[0] / 384)))
    cont_timestamps = np.zeros(int(cont_raw.shape[0] / 384))

    for i, sample in enumerate(TTL_samples):
        ind = sample - cont_start_sample
        cont_samples[ind] = sample
        cont_timestamps[ind] = TTL_timestamps[i]
        if i == 0:
            cont_timestamps[:ind] = np.linspace(
                TTL_timestamps[i] - (1 / 30000. * len(cont_timestamps[:ind - 1])) + 1 / 30000., TTL_timestamps[i],
                len(cont_timestamps[:ind]))
            prev_ind = ind
        else:
            cont_timestamps[prev_ind:ind] = np.linspace(cont_timestamps[prev_ind] + 1 / 30000., TTL_timestamps[i],
                                                        len(cont_timestamps[prev_ind:ind]))
            prev_ind = ind
    cont_timestamps[ind:] = np.linspace(TTL_timestamps[i] + 1 / 30000.,
                                        TTL_timestamps[i] + len(cont_timestamps[ind:]) * 1 / 30000.,
                                        len(cont_timestamps[ind:]))

    if not os.path.exists(os.path.join(directory, 'new_timestamps')):
        os.mkdir(os.path.join(directory, 'new_timestamps'))
    np.save(open(os.path.join(directory, 'new_timestamps', 'sample_numbers.npy'), 'wb'), cont_samples.astype(np.int64))
    np.save(open(os.path.join(directory, 'new_timestamps', 'timestamps.npy'), 'wb'), cont_timestamps.astype(np.float64))


def make_spike_secs(probe_folder):
    """
    Make spike_secs.npy, a file containing spike times along seconds

    :param probe_folder: string path to the probe folder

    :return: None, will save file in directory
    """

    c = np.load(os.path.join(probe_folder,'spike_times.npy'))
    try:
        a = np.load(os.path.join(probe_folder,'timestamps.npy'))
    except:
        try:
            a = np.load(os.path.join(probe_folder, 'new_timestamps', 'timestamps.npy'))
        except:
            try:
                print('could not find timestamps.npy, trying to recreate from the sync TTLs for '+probe_folder)
                recreate_probe_timestamps_from_TTL(probe_folder)
                a = np.load(os.path.join(probe_folder,'new_timestamps','timestamps.npy'))
            except Exception as e:
                print('could not find timestamps.npy')
                raise e

    try:
        spike_secs = a[c.flatten()[np.where(c.flatten()<a.shape[0])]]
        print('shape of spike times annd timestamps not compatible, check above and investigate.')
        np.save(open(os.path.join(probe_folder, 'spike_secs.npy'), 'wb'), spike_secs)
    except Exception as e:
        print("Error making spike_secs.npy!")
        print(np.shape(a))
        print(np.shape(c.flatten()))
        print(np.shape(c))
        raise e


def multi_load_unit_data(recording_folder, probe_names=['A', 'B', 'C', 'D'], probe_depths=[3840, 3840, 3840, 3840],
                         spikes_filename='spike_secs.npy', aligned=True):
    """
    Load multiple units
    """

    folder_paths = glob.glob(os.path.join(recording_folder, '*imec*'))
    if len(folder_paths) > 0:
        spikes_filename = 'spike_secs.npy'
    else:
        folder_paths = glob.glob(os.path.join(recording_folder, '*AP*'))
        if len(folder_paths) > 0:
            for probe_folder in folder_paths: make_spike_secs(probe_folder)
        else:
            print('did not find any recordings in ' + recording_folder + '')
            return
    return pd.concat([load_unit_data(folder, probe_name=probe_names[i], probe_depth=probe_depths[i],
                                     spikes_filename=spikes_filename, aligned=True, df=True) for i, folder in
                      enumerate(folder_paths)], ignore_index=True)
