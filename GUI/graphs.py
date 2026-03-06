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


#functions
def get_good_neurons(db_path, single_units, isi_type, isi_cutoff, session_id):
    """
    Returns good neuron IDs sorted by unit_loc_y (dorsal → ventral).
    Larger y = more dorsal → plotted higher.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if single_units == "True":
        cursor.execute(f"""
            SELECT unit_id, unit_loc_y
            FROM neurons
            WHERE {isi_type} < ?
              AND session_id = ?
              AND label = ?
              AND unit_loc_y IS NOT NULL
            ORDER BY unit_loc_y DESC
        """, (isi_cutoff, session_id, 'auditory'))

        results = cursor.fetchall()
        conn.close()

        # results already sorted dorsal (high y) to ventral (low y)
        return [row[0] for row in results]

    conn.close()
    return []

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
    stim_lib_path = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'

    for s, stim in enumerate(np.unique(stimuli_arr['Stimuli Type'])):
        # --- create subplots based on number of neurons (n+2 plots) ---
        fig, ax = plt.subplots(
            nrows=len(neurons)+2,
            ncols=2,
            figsize=(10, 3*len(neurons)+2),
            gridspec_kw={"width_ratios": [5, 1]}
        )
        for r in range(1, ax.shape[0]):
            ax[r, 0].sharex(ax[0, 0])

        # --- get stimulus ---
        stim_path = os.path.join(stim_lib_path, f'{stim}.wav')
        sampling_rate, data = wav.read(stim_path)
        data = data.astype(float)
        data /= np.max(np.abs(data))
        duration = data.shape[0] / sampling_rate
        t = np.linspace(0, duration, data.shape[0])
        ax[0,0].plot(t+1, data)

        # --- plot raster for each neuron separately ---
        curr_stim = stimuli_arr[stimuli_arr['Stimuli Type'] == stim].reset_index(drop=True)
        for u, unit in enumerate(neurons):
            spikes = np.array(data_dict[f'unit_{unit}'])/30000

            all_spks = []
            for r, row in curr_stim.iterrows():
                start_time = row['Start Time']-1
                end_time = row['End Time']+1
                mask = (spikes >= start_time) & (spikes < end_time)

                trial_spks = spikes[mask]-start_time
                ax[u+1,0].vlines(trial_spks, ymin=r, ymax=r+0.5)
                ax[u+1,0].set_ylabel(f'Cluster {unit}\n\nTrials')
                all_spks.extend(trial_spks)

            bins = np.arange(0, np.max(curr_stim['Duration']) + 10 / 1000 + 2, 10 / 1000)
            counts, edges = np.histogram(sorted(all_spks), bins = bins)
            counts = counts / len(curr_stim) * 1000 / 10
            ax[-1,0].plot(edges[:-1], counts)

            # --- put waveforms next to units ---
            with h5py.File(waveforms_path, "r") as f:
                mean = f['units'][f'unit_{unit}']['mean'][:]
                sd = f['units'][f'unit_{unit}']['sd'][:]

                x = np.arange(len(mean))
                ax[u+1,1].plot(x, mean, color='black')
                ax[u+1,1].fill_between(x, mean-sd, mean+sd, alpha=0.2)

        if len(neurons) > 0:
            ax[-2, 1].hlines(
                y=min(mean) - 150,
                xmin=max(x) - 30,  # 1 ms --> samplerate/1000
                xmax=max(x),
                color='black',
                linestyle='-',
                linewidth=2
            )
            ax[-2, 1].text(
                x=(max(x)),
                y=min(mean) - 100,
                s=f'{1} ms',
                color='black',
                ha='right',
                va='center',
                size=10
            )


            # --- format plots ---
            # if args.single_units=="True":
            #     plt.suptitle(f'Single Unit Responses to {stim}')
            # else:
            #     plt.suptitle(f'Multiunit Responses to {stim}')
            ax[0,0].spines['top'].set_visible(False)
            ax[0,0].spines['bottom'].set_visible(False)
            ax[0,0].spines['right'].set_visible(False)
            ax[0,0].set_yticks([-1,0,1])
            ax[0,0].set_ylabel("Stimulus Amplitude (a.u.)")
            for a in ax[:, 1]:
                a.spines['top'].set_visible(False)
                a.spines['bottom'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.spines['left'].set_visible(False)
                a.set_xticks([])
                a.set_yticks([])
            for a in ax[1:-1,0]:
                a.spines['top'].set_visible(False)
                a.spines['bottom'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.set_xticks([])

            for a in ax[1:-1, 1]:
                a.spines['top'].set_visible(False)
                a.spines['bottom'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.set_xticks([])

            ax[-1,0].spines['top'].set_visible(False)
            ax[-1,0].spines['right'].set_visible(False)
            # ax[-1,1].spines['top'].set_visible(False)
            # ax[-1,1].spines['bottom'].set_visible(False)
            # ax[-1,1].spines['right'].set_visible(False)
            # ax[-1,1].spines['left'].set_visible(False)
            plt.tight_layout()

            print(args.recording_path)
            bird_id = os.path.join(args.recording_path).split('\\')[-1].split(' ')[0]
            print(f'Bird ID: {bird_id}')
            brain_area = args.recording_path.split(', ')[1]
            rec_num = args.recording_path.split('#')[1].split(' ')[0]
            print(f'Brain area: {brain_area}')
            bird_rasters_path = os.path.join(args.recording_path, '..', '..', '..', 'Raster Plots', bird_id)
            print(f'Bird rasters path: {bird_rasters_path}')

            if not os.path.exists(bird_rasters_path):
                os.makedirs(bird_rasters_path)
            if not os.path.exists(os.path.join(bird_rasters_path, brain_area)):
                os.makedirs(os.path.join(bird_rasters_path, brain_area))

            plt.savefig(fr"R:\Data\RhythmPerception\Neural Recordings\Raster Plots\{bird_id}\{brain_area}\{bird_id}_{brain_area}_rec{rec_num}_{stim}.png")
        else:
            print("no neurons to show.")
    print("done plotting data.")

# ------------------------------------------------------------------ #
#                     Plot Probe                                     #
# ------------------------------------------------------------------ #
recording_name = args.recording_path.split("\\")[-1]
temp_neural_files_path = r'C:\Users\tmerri03\Desktop\Temp Neural Files'
if os.path.exists(os.path.join(temp_neural_files_path, recording_name)):
    analyzer_path = os.path.join(args.recording_path, 'sorting', 'analyzer_TDC_binary')
    sorting_path = os.path.join(args.recording_path, 'sorting', 'sorting_TDC')
    curation_fp = os.path.join(analyzer_path, 'spikeinterface_gui', 'curation_data.json')
    recording_path = os.path.join(temp_neural_files_path, recording_name, 'sorting', 'preprocessed')

    # print(f'recording_path: {recording_name}')

    analyzer_TDC = si.load(analyzer_path)
    sorting_TDC = si.load(sorting_path)
    recording = si.load(recording_path)

    print('loading analyzer...')
    if os.path.exists(curation_fp):
        print("     data has been manually curated.")
        curation = sc.load_curation(curation_fp)
        clean_sorting = sc.apply_curation(sorting_TDC, curation_dict_or_model=curation)
        print("     applying curation...")
        analyzer = sc.apply_curation(analyzer_TDC, curation_dict_or_model=curation)

        if not analyzer.has_recording():
            print("     recording missing from analyzer — reattaching")

            analyzer = si.create_sorting_analyzer(
                sorting=analyzer.sorting,
                recording=recording,
                folder=analyzer_path,
                format="binary_folder",
                overwrite=True
            )

    else:
        print("     no manual curation has been found.")
        analyzer = analyzer_TDC

    if analyzer.get_extension('random_spikes') is None:
        analyzer.compute('random_spikes')
    if analyzer.get_extension('waveforms') is None:
        analyzer.compute('waveforms')
    if analyzer.get_extension('templates') is None:
        analyzer.compute('templates')
    if analyzer.get_extension("spike_locations") is None:
        analyzer.compute("spike_locations")
    if analyzer.get_extension("unit_locations") is None:
        analyzer.compute("unit_locations")

    unit_locations = pd.DataFrame(analyzer.get_extension("unit_locations").get_data())
    unit_ids = analyzer.sorting.unit_ids
    unit_locations.index = unit_ids
    unit_locations_good = unit_locations.loc[neurons]

    x_units = unit_locations_good.iloc[:, 0].values
    y_units = unit_locations_good.iloc[:, 1].values
    unit_ids_good = unit_locations_good.index.values

    print('plotting probe and good units...')
    fig, ax = plt.subplots(figsize=(4, 8))

    # probe
    probe = recording.get_probe()
    contact_positions = probe.contact_positions

    ax.scatter(
        contact_positions[:, 0],
        contact_positions[:, 1],
        s=10,
        c='k',
        alpha=0.5,
        label='Probe contacts'
    )

    # good units
    ax.scatter(
        x_units,
        y_units,
        s=70,
        c='red',
        edgecolor='white',
        zorder=3,
        label='Good units'
    )

    # labels
    for uid, x, y in zip(unit_ids_good, x_units, y_units):
        ax.text(
            float(x),
            float(y),
            str(uid),
            fontsize=8,
            ha='center',
            va='center',
            zorder=4,
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
        )

    ax.set_aspect('equal')

    # ax.invert_yaxis()
    plt.savefig(fr"R:\Data\RhythmPerception\Neural Recordings\Raster Plots\{bird_id}\{brain_area}\{bird_id}_{brain_area}_rec{rec_num}_probe.svg", format='svg')
else:
    print("cannot find local copy of pre-processed data. cannot plot probe.")