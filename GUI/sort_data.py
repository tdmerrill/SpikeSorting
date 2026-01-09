import os
import shutil
import sys
import argparse, time, pprint

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.full as si
from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe
import numpy as np
import pandas as pd
import json, os
import matplotlib.pyplot as plt

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
                f"\r[{bar}] {progress*100:6.2f}%  "
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

if __name__ == '__main__':
    # ---- get input information ----
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--chanmap")
    parser.add_argument("--sorter")

    args = parser.parse_args()

    # ---- copy files locally for processing ----
    copy_dir = args.data
    rec_name = os.path.abspath(os.path.join(copy_dir, '..')).split('\\')[-1]
    dest_dir = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Temp Neural Files', rec_name, 'unfiltered')
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
    recording_cmr = si.common_reference(recording_f, reference='global', operator='median')

    if not os.path.exists(preprocess_save_folder):
        recording_preprocessed = recording_cmr.save(format='binary', folder=preprocess_save_folder, overwrite=False)
    else:
        print('Preprocessed data already exists, manually delete and rerun to override.')
        recording_preprocessed = si.load(preprocess_save_folder)
    print(recording_preprocessed)

    plt.savefig(os.path.join(preprocess_save_folder, 'probe_configuration.png'))
    print("Saved probe to probe_configuration.png")

    print("")
    print("---- Sorter Parameters ----")
    pprint.pprint(ss.get_default_sorter_params(args.sorter))

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

    from spikeinterface_gui import run_mainwindow
    sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
    run_mainwindow(sorting_analyzer, mode='desktop', curation=True)

    # ---- copy files over if the first time sorting ----
    destination_dir = os.path.join(args.data, '..', 'sorting')
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print("Copying sorting output to R:")
        copy_directory_with_progress(sorting_folder, os.path.join(destination_dir, 'sorting_TDC'))
        print("Copying curation output to R:")
        copy_directory_with_progress(analyzer_folder, os.path.join(destination_dir, 'analyzer_TDC_binary'))
        print("Success!")

    # ---- check if there are sorting differences ----
    if os.path.exists(destination_dir):
        print("Base directory already exists!")
        print("Checking for sorting output...")
        if os.path.exists(os.path.join(destination_dir, 'sorting_TDC')):
            print("Sorting output already exists!")
        else:
            copy_directory_with_progress(sorting_folder, os.path.join(destination_dir, 'sorting_TDC'))

        print("Checking for curation output...")
        if os.path.exists(os.path.join(destination_dir, 'analyzer_TDC_binary')):
            print("Curation output already exists -- checking for differences in curation.")
            gui_output_path_curr = os.path.join(destination_dir, 'analyzer_TDC_binary', 'spikeinterface_gui', 'curation_data.json')
            gui_output_path_new = os.path.join(analyzer_folder, 'spikeinterface_gui', 'curation_data.json')
            if os.path.exists(gui_output_path_curr):
                if json_equal(gui_output_path_new, gui_output_path_curr):
                    print("Curation did not change!")
                else:
                    print("Curation is different - removing old file!")
                    os.remove(gui_output_path_curr)
                    shutil.copy2(gui_output_path_new, gui_output_path_curr)
            else:
                # if the analyzer folder is present, but not gui output
                print("Analyzer folder present, but curation output missing -- replacing with new info!")
                if os.path.exists(gui_output_path_curr) and os.path.exists(gui_output_path_new):
                    os.remove(gui_output_path_curr)
                    shutil.copy2(gui_output_path_new, gui_output_path_curr)

        else:
            print("Curation output not found -- copying to R:")
            copy_directory_with_progress(analyzer_folder, os.path.join(destination_dir, 'analyzer_TDC_binary'))

    print(time.time()-start_time)