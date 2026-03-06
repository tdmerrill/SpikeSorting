#imports
import time, argparse, pprint
import os, json
import numpy as np
import pandas as pd
import h5py
import sqlite3
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import matplotlib
import spikeinterface.full as si
import spikeinterface.curation as sc
import re

def get_good_neurons(db_path, single_units, isi_type, isi_cutoff, session_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if single_units=="True":
        cursor.execute(f"""
                SELECT unit_id
                FROM neurons
                WHERE {isi_type} < ? AND session_id = ? AND label = ?
        """, (isi_cutoff,session_id, 'auditory'))
        neurons = cursor.fetchall()

        return [x[0] for x in np.array(neurons)]


def classify_stim(stim_name):
    stim_upper = stim_name.upper()

    # ---------------- SONGS ----------------
    if any(x in stim_upper for x in ["BOS", "HET", "CON"]):
        row_priority = 0  # songs first

        if "BOS" in stim_upper:
            stim_priority = 0
        elif "HET" in stim_upper:
            stim_priority = 1
        elif "CON" in stim_upper:
            stim_priority = 2
        else:
            stim_priority = 3

        tempo = np.nan


    # ---------------- SOUND A ----------------
    elif "ZF A" in stim_upper:
        row_priority = 1

        tempo_match = re.search(r"(\d+)MS", stim_upper)
        tempo = float(tempo_match.group(1)) if tempo_match else np.nan

        if "REG" in stim_upper:
            stim_priority = 0
        elif "IRREG" in stim_upper:
            stim_priority = 1
        else:
            stim_priority = 2


    # ---------------- SOUND E ----------------
    elif "ZF E" in stim_upper:
        row_priority = 2

        tempo_match = re.search(r"(\d+)MS", stim_upper)
        tempo = float(tempo_match.group(1)) if tempo_match else np.nan

        if "REG" in stim_upper:
            stim_priority = 0
        elif "IRREG" in stim_upper:
            stim_priority = 1
        else:
            stim_priority = 2


    # ---------------- OTHER ----------------
    else:
        row_priority = 99
        stim_priority = 99
        tempo = np.nan

    return row_priority, stim_priority, tempo

if __name__ == '__main__':
    # ---- get input information ----
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--recording_path", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--single_units", required=True)
    parser.add_argument("--isi_cutoff", type=float, default=100)
    parser.add_argument('--isi_type', type=str, default='manual_isi_1')
    parser.add_argument("--session_id", type=str, required=True)

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #                     Load Spikes & Stim                             #
    # ------------------------------------------------------------------ #

    print('Loading spikes...')
    spikes_path = os.path.join(args.recording_path, 'spikes.h5')
    stimuli_path = os.path.join(args.recording_path, 'stimuli.json')
    waveforms_path = os.path.join(args.recording_path, 'waveforms.h5')

    stimuli_arr = pd.read_json(stimuli_path)

    neurons = get_good_neurons(args.db_path, args.single_units, args.isi_type, args.isi_cutoff, args.session_id)

    data_dict = {}
    with h5py.File(spikes_path, "r") as f:
        for key in f.keys():
            if int(key.split('_')[-1]) in neurons:
                print(f'Loading data for neuron {key.split("_")[-1]}')
                data_dict[key] = f[key][:]

    print('done loading data.')

    # ------------------------------------------------------------------ #
    #                     Plot Data                                      #
    # ------------------------------------------------------------------ #
    print('test')
    stim_lib_path = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
    stimuli_arr["stim_class"] = stimuli_arr["Stimuli Type"].apply(classify_stim)
    df = stimuli_arr.copy()

    df[["row_priority", "stim_priority", "tempo"]] = (
        df["Stimuli Type"]
        .apply(lambda x: pd.Series(classify_stim(x)))
    )

    df = df.sort_values(
        ["row_priority", "stim_priority", "tempo"]
    )

    print(df.head())
    print('all done')