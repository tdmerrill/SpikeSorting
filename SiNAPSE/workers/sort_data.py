import os
import shutil
import sys
import argparse, time, pprint

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.full as si
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc
from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe
import numpy as np
import pandas as pd
import json, os
import matplotlib.pyplot as plt
import sqlite3, h5py
from scipy.signal import resample_poly
from pathlib import Path

def copy_directory_with_progress(src_path, dst_path):
    """
    Copy the entire directory at src_path into dst_path.
    Shows a terminal progress bar during copy.
    """

    if not os.path.isdir(src_path):
        raise ValueError(f"Source path does not exist or is not a directory: {src_path}")

    # Remove existing destination
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)

    # Count files first
    total_files = 0
    for root, dirs, files in os.walk(src_path):
        total_files += len(files)

    if total_files == 0:
        shutil.copytree(src_path, dst_path)
        print("No files to copy. Directory copied.")
        return

    copied_files = 0

    # Create root folder
    os.makedirs(dst_path, exist_ok=True)

    print(f"Copying directory:\n  from: {src_path}\n    to: {dst_path}\n")

    # Copy manually for progress reporting
    for root, dirs, files in os.walk(src_path):

        # Create directories
        rel = os.path.relpath(root, src_path)
        dest_root = os.path.join(dst_path, rel)
        os.makedirs(dest_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_root, file)

            shutil.copy2(src_file, dst_file)

            copied_files += 1
            progress = copied_files / total_files
            bar_len = 40
            filled = int(bar_len * progress)
            bar = "#" * filled + "-" * (bar_len - filled)

            # Print progress bar
            sys.stdout.write(
                f"\r[{bar}] {progress * 100:6.2f}%  "
                f"({copied_files}/{total_files} files)"
            )
            sys.stdout.flush()

    print("\nCopy complete!")

def json_equal(file1, file2):
    with open(file1, 'r') as f1:
        data1 = json.load(f1)
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    return data1 == data2

def get_unit_locs(analyzer):
    if analyzer.get_extension("unit_locations") is None:
        analyzer.compute("unit_locations")
    unit_locations = analyzer.get_extension("unit_locations").get_data()
    locs = pd.DataFrame(unit_locations)
    return locs[0], locs[1]

def get_log_file_and_stimulus_windows(root_dir, samplerate=30000):
    filtered_rec = os.path.join(root_dir, "filtered")
    unfiltered_rec = os.path.join(root_dir, "unfiltered")

    # --- load log file ---
    csv_files = [
        os.path.join(filtered_rec, f)
        for f in os.listdir(filtered_rec)
    ]
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV log file found in filtered/")
    if len(csv_files) > 1:
        print("Warning: multiple CSV files found, using first one")

    log_path = csv_files[0]
    df = pd.read_csv(log_path, header=1)

    # --- get recording start time from sync_messages ---
    step1 = os.path.join(unfiltered_rec, os.listdir(unfiltered_rec)[0])
    step2 = os.path.join(step1, os.listdir(step1)[0])
    step3 = os.path.join(step2, os.listdir(step2)[0])
    step4 = os.path.join(step3, os.listdir(step3)[0])
    sync_messages_path = os.path.join(step4, 'sync_messages.txt')

    print(f'Sync Messages Filepath: {sync_messages_path}')

    recording_start_time = None
    try:
        with open(sync_messages_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                recording_start_time = int(parts[-1])  # last line wins
    except FileNotFoundError:
        raise FileNotFoundError(f"sync_messages.txt not found at {sync_messages_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading start time: {e}")

    print(f"Start Time (samples): {recording_start_time}")

    # --- build session and recording ---
    from open_ephys.analysis import Session
    session_path = os.path.join(unfiltered_rec, os.listdir(unfiltered_rec)[0])
    session = Session(session_path)
    recording = session.recordnodes[0].recordings[0]
    sample_time = ((recording.events.sample_number) - int(recording_start_time)) / 30000

    diffs = np.diff(sample_time)
    mask = (diffs > 0.15) & (diffs < 4.9)  # filter between 0.15 and 5 seconds
    start_time = sample_time[:-1][mask].reset_index(drop=True)
    end_time = sample_time[1:][mask].reset_index(drop=True)
    durations = diffs[mask]

    # ---- and put them all into a dataframe ----
    stimuli_arr = pd.DataFrame(columns=['Start Time', 'End Time', 'Duration', 'Stimuli Type'])
    stimuli_arr['Start Time'] = start_time
    stimuli_arr['End Time'] = end_time
    stimuli_arr['Duration'] = durations
    stimuli_arr['Start Time'] = stimuli_arr['Start Time']
    stimuli_arr['End Time'] = stimuli_arr['End Time']
    stimuli_arr['Stimuli Type'] = np.array(df["Stimulus"])

    stimuli_arr.to_json(os.path.join(root_dir, 'stimuli.json'), orient='records', indent=2)

    return unfiltered_rec

if __name__ == '__main__':
    # ---- get input information ----
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--chanmap")
    parser.add_argument("--sorter")
    parser.add_argument('--destination', required=True)

    args = parser.parse_args()

    # ---- copy files locally for processing ----
    copy_dir = args.data

    # rec_name = os.path.abspath(os.path.join(copy_dir, '..')).split('\\')[-1]
    rec_name = Path(os.path.abspath(os.path.join(copy_dir, '..'))).name
    print(rec_name)
    # dest_dir = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Temp Neural Files', rec_name, 'unfiltered')
    dest_dir = os.path.join(args.destination, rec_name)
    print(dest_dir)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        copy_directory_with_progress(copy_dir, dest_dir)
    else:
        print("Local copy already exists!")

    # ---- find openephys folder ----
    oe_path = os.path.join(dest_dir, os.listdir(dest_dir)[0])

    # ---- initiate sorting ----
    recording = se.read_openephys(oe_path, stream_id='0')
    channel_ids = recording.get_channel_ids()
    fs = recording.get_sampling_frequency()
    num_chan = recording.get_num_channels()
    num_seg = recording.get_num_segments()

    print("Channel ids:", channel_ids)
    print("Sampling frequency:", fs)
    print("Number of channels:", num_chan)
    print("Number of segments:", num_seg)

    my_probe = get_probe(manufacturer="cambridgeneurotech", probe_name=args.probe)
    my_probe.set_device_channel_indices(json.loads(args.chanmap))
    recording_probe = recording.set_probe(my_probe, group_mode='by_shank')
    plot_probe(recording_probe.get_probe())

    preprocess_save_folder = os.path.join(oe_path, '..', '..', 'sorting', 'preprocessed')
    recording_cmr = recording_probe
    recording_f = si.bandpass_filter(recording_probe, freq_min=300, freq_max=10000)

    removed_ch_path = os.path.join(args.data, '..', 'removed_ch.json')

    bad_channels = []
    if os.path.exists(removed_ch_path):
        with open(removed_ch_path, 'r') as f:
            j = json.load(f)
        bad_channels = np.array(j['Removed Channels']['channels'])
        print(f'Removed channels (from probe): {bad_channels}')
        ref_channels = [c for c in channel_ids if c not in bad_channels]

    if len(bad_channels) > 0:
        recording_cmr = si.common_reference(
            recording_f,
            reference='global',
            operator='median',
            ref_channel_ids=ref_channels
        )
        print('CMR filter with removed channels.')
    else:
        recording_cmr = si.common_reference(recording_f, reference='global', operator='median')

    if len(bad_channels) > 0:
        recording_final = recording_cmr.remove_channels(remove_channel_ids=bad_channels)
        print('removed channels from recording_final')
    else:
        recording_final = recording_cmr

    if not os.path.exists(preprocess_save_folder):
        recording_preprocessed = recording_final.save(format='binary', folder=preprocess_save_folder, overwrite=False)
        print("Final channel ids:", recording_preprocessed.get_channel_ids())
        print("Final number of channels:", recording_preprocessed.get_num_channels())
    else:
        print('Preprocessed data already exists, manually delete and rerun to override.')
        recording_preprocessed = si.load(preprocess_save_folder)
    print(recording_preprocessed)

    plt.savefig(os.path.join(preprocess_save_folder, 'probe_configuration.png'))
    print("Saved probe to probe_configuration.png")

    # print("")
    # print("---- Sorter Parameters ----")
    # pprint.pprint(ss.get_default_sorter_params(args.sorter))

    analyzer_folder = os.path.join(preprocess_save_folder, '..', 'analyzer_TDC_binary')
    sorting_folder = os.path.join(preprocess_save_folder, '..', 'sorting_TDC')
    if os.path.exists(sorting_folder):
        print("Loading existing sorting...")
        sorting_TDC = si.load(sorting_folder)
    else:
        print("Running sorter...")
        sorting_TDC = ss.run_sorter(
            sorter_name=args.sorter,
            recording=recording_preprocessed,
            folder=sorting_folder,
        )
    if os.path.exists(analyzer_folder):
        print("Loading existing analyzer...")
        analyzer_TDC = si.load_sorting_analyzer(analyzer_folder)
    else:
        print("Creating analyzer...")
        analyzer_TDC = si.create_sorting_analyzer(
            sorting=sorting_TDC,
            recording=recording_preprocessed,
            format='binary_folder',
            folder=analyzer_folder
        )

    print(analyzer_TDC)

    if analyzer_TDC.get_extension("random_spikes") is None:
        analyzer_TDC.compute("random_spikes")
    if analyzer_TDC.get_extension("waveforms") is None:
        analyzer_TDC.compute("waveforms")
    if analyzer_TDC.get_extension("noise_levels") is None:
        analyzer_TDC.compute("noise_levels")
    if analyzer_TDC.get_extension("templates") is None:
        analyzer_TDC.compute("templates")
    if analyzer_TDC.get_extension("spike_amplitudes") is None:
        analyzer_TDC.compute("spike_amplitudes")
    if analyzer_TDC.get_extension("spike_locations") is None:
        analyzer_TDC.compute("spike_locations")
    if analyzer_TDC.get_extension("unit_locations") is None:
        analyzer_TDC.compute("unit_locations")

    from spikeinterface_gui import run_mainwindow

    sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
    run_mainwindow(sorting_analyzer, mode='desktop', curation=True)
    test = 'FALSE'
    # ---- apply curation from GUI ----
    curation_fp = os.path.join(analyzer_folder, 'spikeinterface_gui', 'curation_data.json')
    if os.path.exists(curation_fp):
        print("Data has been manually curated.")
        curation = sc.load_curation(curation_fp)
        clean_sorting = sc.apply_curation(sorting_TDC, curation_dict_or_model=curation)
        print("Applying curation...")
        analyzer = sc.apply_curation(sorting_analyzer, curation_dict_or_model=curation)
    else:
        print("No manual curation has been found.")
        analyzer = analyzer_TDC

    # # ---- find spike widths ----
    # we = analyzer.get_extension("waveforms")
    # fs = analyzer.sampling_frequency
    # dt_ms = 1000 / fs
    # all_wfs = we.get_data()  # dict: unit_id -> waveforms
    # spike_widths_ms = {}
    #
    # for unit_id in analyzer.unit_ids:
    #     wfs = all_wfs[unit_id]
    #     if wfs.ndim == 3:
    #         # (n_spikes, n_samples, n_channels)
    #         mean_wf = wfs.mean(axis=0)
    #         chan = np.argmin(mean_wf.min(axis=0))
    #         wf = mean_wf[:, chan]
    #     else:
    #         mean_wf = wfs.mean(axis=0)
    #         wf = mean_wf
    #
    #     trough_idx = np.argmin(wf)
    #     peak_idx = trough_idx + np.argmax(wf[trough_idx:])
    #
    #     width_ms = (peak_idx - trough_idx) * dt_ms
    #     spike_widths_ms[unit_id] = width_ms

    # ---- run additional analysis to find quality metrics
    sqm.compute_quality_metrics(
        analyzer,
        metric_names=[
            'snr',
            'firing_rate',
            'isi_violation',
            'presence_ratio',
            'sliding_rp_violation',
            'drift',
            'amplitude_median',
            'amplitude_cv',
            'noise_cutoff'
        ],
        qm_params={"refractory_period_ms": 1}
    )
    qm = analyzer.get_extension("quality_metrics").get_data()
    # qm['spike_width'] = pd.Series(spike_widths_ms)
    # print(qm.columns)
    print(f'There are {len(qm)} neurons!')