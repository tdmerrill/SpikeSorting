import os
import shutil
import tempfile
import json
import subprocess, threading
import sqlite3
import numpy as np
import pandas as pd
from numpy.ma.core import masked
from scipy.ndimage import gaussian_filter1d
import h5py
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()
import threading

from signals import signals

class MyFunctions:
    def __init__(self):
        print("initializing my functions")

        self.single_units = None
        self.brain_region = 'NCM'
        signals.send_neuron_filters.connect(self.set_neuron_filters)

        self.db_path = r"C:\Users\tmerri03\Desktop\Temp Neural Files\neurons.db"
        self.data_root = r'R:\Data\RhythmPerception\Neural Recordings\Recordings'

    def curate_responses(self, isi_cutoff=0.1):
        cmd = [
            'python',
            'GUI/LabelAuditoryNeurons.py',
            ]
        p = subprocess.Popen(cmd)

    def label_brain_area(self, isi_cutoff=1):
        # --- get non-isi violation units ---
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT id, session_id, unit_id
                       FROM neurons
                       WHERE manual_isi_1 < ?
                         AND spike_width_pp IS NOT NULL
                         AND spike_width_hw IS NOT NULL
                         AND label = ?
                       ''', (1, 'auditory'))
        n = cursor.fetchall()
        conn.close()

        if len(n) > 0:
            tabs = {}
            for id, session_id, unit_id in n:
                bird_id = session_id.split(' ')[0]
                if bird_id not in tabs.keys():
                    tabs[bird_id] = {}
                if session_id not in tabs[bird_id].keys():
                    tabs[bird_id][session_id] = []
                tabs[bird_id][session_id].append(f'neuron_{unit_id}')

            with open("neurons_to_label.json", "w") as f:
                json.dump(tabs, f)

            subprocess.Popen(["streamlit", "run", r"C:\Users\tmerri03\PycharmProjects\SpikeSorting\GUI\label_brain_area.py"])

        else:
            print('no neurons found!')

    def plot_spike_width(self, single_units, isi_cutoff=1):
        areas = ['NCM', 'Field L', 'Area X', 'HVC', 'CM']
        area_dict_pp, area_dict_hw = {}, {}
        for a in areas:
            if a not in area_dict_hw:
                area_dict_hw[a] = []
            if a not in area_dict_pp:
                area_dict_pp[a] = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if single_units:
            # --- get non-isi violation units ---
            cursor.execute('''
                           SELECT session_id, spike_width_pp, spike_width_hw
                           FROM neurons
                           WHERE manual_isi_1 < ?
                           AND spike_width_pp IS NOT NULL
                           AND spike_width_hw IS NOT NULL
                           AND label = ?
                           ''', (isi_cutoff,'auditory'))
            FRs = cursor.fetchall()

            for i, item in enumerate(FRs):
                area = item[0].split(",")[1]
                area_dict_hw[area.strip()].append(item[2])
                area_dict_pp[area.strip()].append(item[1])

            # --- histogram firing rates ---


            # bins = np.linspace(0, 1, 33)  # 20 bins, adjust as needed
            fig, ax = plt.subplots(1,2, figsize = (10,5))

            all_values_hw, all_values_pp = [], []
            for v in area_dict_hw.values():
                # make sure we only keep non-empty values
                clean_v = [float(x) for x in v if x not in (None, '', 'NaN')]
                all_values_hw.extend(clean_v)
            all_values = np.array(all_values_hw, dtype=float)

            for area, values in area_dict_hw.items():
                bins = 30
                counts, edges = np.histogram(values, bins=bins, range=(0, 1.0))
                centers = (edges[:-1] + edges[1:]) / 2  # histogram bin centers
                y = counts / counts.sum()
                y_smooth = gaussian_filter1d(y, sigma=1.2)

                ax[0].plot(centers, y_smooth, label=area, linewidth=2)
            ax[0].set_title("Spike Width - Half Width")
            ax[0].legend()

            for v in area_dict_pp.values():
                # make sure we only keep non-empty values
                clean_v = [float(x) for x in v if x not in (None, '', 'NaN')]
                all_values_hw.extend(clean_v)
            all_values = np.array(all_values_hw, dtype=float)

            for area, values in area_dict_pp.items():
                bins = 30
                counts, edges = np.histogram(values, bins=bins, range=(0, 1.0))
                centers = (edges[:-1] + edges[1:]) / 2  # histogram bin centers
                y = counts / counts.sum()
                y_smooth = gaussian_filter1d(y, sigma=1.2)

                ax[1].plot(centers, y_smooth, label=area, linewidth=2)
            ax[1].set_title("Spike Width - Peak to Peak")

            plt.xlabel("Spike Width (ms)")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_FR(self, single_units, isi_cutoff=1):
        areas = ['NCM', 'Field L', 'Area X', 'HVC', 'CM']
        area_dict = {}
        for a in areas:
            if a not in area_dict:
                area_dict[a] = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if single_units:
            # --- get non-isi violation units ---
            cursor.execute('''
                    SELECT session_id, firing_rate
                    FROM neurons
                    WHERE manual_isi_1 < ?
                        AND label = ?
            ''', (isi_cutoff,'auditory'))
            FRs = cursor.fetchall()

            for i, item in enumerate(FRs):
                area = item[0].split(",")[1]
                area_dict[area.strip()].append(item[1])

            # --- histogram firing rates ---
            all_values = np.concatenate(list(area_dict.values()))
            max_val = all_values.max()

            bins = np.linspace(0, max_val, 500)  # 20 bins, adjust as needed
            plt.figure(figsize=(8, 5))

            for area, values in area_dict.items():
                counts, edges = np.histogram(values, bins=bins)
                counts = counts / sum(counts)
                y_smooth = gaussian_filter1d(counts, sigma=2)
                centers = (edges[:-1] + edges[1:]) / 2  # histogram bin centers
                plt.plot(centers, y_smooth, label=area, linewidth=2)

            plt.title("Firing Rate by Brain Region")
            plt.xlabel("Firing rate (Hz)")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 5))

            for area, values in area_dict.items():
                counts, edges = np.histogram(values, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2  # histogram bin centers
                y_smooth = gaussian_filter1d(counts, sigma=1.2)

                plt.plot(centers, y_smooth / sum(y_smooth), label=area, linewidth=2)

            plt.xlim(0,5)
            plt.title("Firing Rate by Brain Region")
            plt.xlabel("Firing rate (Hz)")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_drift(self, single_units, isi_cutoff=0.1):
        areas = ['NCM', 'Field L', 'Area X', 'HVC', 'CM']
        area_dict = {}
        for a in areas:
            if a not in area_dict:
                area_dict[a] = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if single_units:
            # --- get non-isi violation units ---
            cursor.execute('''
                    SELECT session_id, drift
                    FROM neurons
                    WHERE isi_violation_ratio < ?
            ''', (isi_cutoff,))
            drifts = cursor.fetchall()

            for i, item in enumerate(drifts):
                area = item[0].split(",")[1]
                area_dict[area.strip()].append(item[1])

            # --- histogram firing rates ---
            all_values = np.concatenate(list(area_dict.values()))
            max_val = all_values.max()

            bins = np.linspace(0, max_val, 50)  # 20 bins, adjust as needed
            plt.figure(figsize=(8, 5))

            for area, values in area_dict.items():
                counts, edges = np.histogram(values, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2  # histogram bin centers
                plt.plot(centers, counts/sum(counts), label=area, linewidth=2)

            plt.title("Drift Rate by Brain Region")
            plt.xlabel("Drift")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def neuron_clicked(self, t):
        bird = t.split(" ")[0]
        recording_path = os.path.join(self.data_root, bird, t)

        cmd = [
            'python',
            'GUI/graphs.py',
            '--recording_path', recording_path,
            '--db_path', self.db_path,
            '--single_units', str(self.single_units),
            '--session_id', t,
        ]
        p = subprocess.Popen(cmd)

    def set_neuron_filters(self, brain_region, single_units):
        self.brain_region = brain_region
        self.single_units = single_units

    def file_clicked(self, path):
        probe, channel_map = self.search_probe(path)
        data_path = os.path.abspath(os.path.join(path, 'unfiltered'))

        cmd = [
            'python',
            'GUI/sort_data.py',
            '--data', data_path,
            '--probe', probe,
            '--chanmap', json.dumps(channel_map),
            "--sorter", "kilosort4"
        ]
        p = subprocess.Popen(cmd)
        threading.Thread(target=self._wait_for_process, args=(p,), daemon=True).start()

    def _wait_for_process(self, p):
        p.wait()
        print("Sorting finished -- updating neuron list.")
        self.update_neuron_list()

    def update_neuron_list(self):
        signals.get_neuron_filters.emit()

        su = "all neurons"
        if self.single_units:
            su = 'single units'
        print(f"Getting {su} in {self.brain_region}.")

        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            cursor = conn.cursor()

            query = """
                    SELECT session_id
                    FROM neurons
                    WHERE LOWER(session_id) LIKE ? \
                    """
            param = f"%{self.brain_region.lower()}%"  # match anywhere in the string

            cursor.execute(query, (param,))
            recordings = [row[0] for row in cursor.fetchall()]

            signals.send_recordings.emit(list(set(recordings)))
        except sqlite3.OperationalError as e:
            print("Database busy or locked, try again later:", e)
            recordings = []
        finally:
            conn.close()


        conn.close()

        self.add_manual_isi()

    def add_manual_isi(self):
        print("adding manual isi where it's missing!")
        db_path = r"C:\Users\tmerri03\Desktop\Temp Neural Files\neurons.db"
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        cur.execute('''
                    SELECT id, session_id, unit_id, spike_file
                    FROM neurons
                    WHERE (
                        manual_isi_0_7 IS NULL
                            OR manual_isi_1 IS NULL
                            OR manual_isi_1_5 IS NULL
                        )
                      AND spike_file IS NOT NULL
                    ''')
        neurons = cur.fetchall()

        if len(neurons) == 0:
            print("No neurons found")
        else:
            for n, neuron in enumerate(neurons):
                id = neuron[0]
                session_id = neuron[1]
                unit_id = neuron[2]
                spike_file = neuron[3]

                print(f'working on neuron {n + 1}/{len(neurons)}')

                data_dict = {}
                with h5py.File(spike_file, "r") as f:
                    for key in f.keys():
                        data_dict[key] = f[key][:]

                spikes = np.array(data_dict[f'unit_{unit_id}']) / 30000
                isi = np.array(np.diff(spikes))
                violation_rate_1_5 = len(isi[isi < 1.5 / 1000]) / len(spikes) * 100
                violation_rate_1 = len(isi[isi < 1 / 1000]) / len(spikes) * 100
                violation_rate_0_7 = len(isi[isi < 0.7 / 1000]) / len(spikes) * 100

                cur.execute(
                    '''
                    UPDATE neurons
                    SET manual_isi_1_5 = ?,
                        manual_isi_1   = ?,
                        manual_isi_0_7 = ?
                    WHERE id = ?
                      AND session_id = ?
                      AND unit_id = ?
                    ''',
                    (violation_rate_1_5, violation_rate_1, violation_rate_0_7, id, session_id, unit_id)
                )
        con.commit()
        con.close()

    def search_probe(self, path):
        if 'P1' in path:
            probe = "ASSY-37-P-1"
        elif 'H4' in path or 'H2' in path:
            probe = "ASSY-37-H4"
        elif 'H6b' in path:
            probe = "ASSY-37-H6b"
        # elif 'E1_1' in path:
        #     probe = "ASSY-1-E-1"
        elif 'E1_2' in path or 'E1' in path:
            probe = "ASSY-37-E-1"
        elif 'H7' in path:
            probe = "ASSY-37-H7b"
        else:
            probe = None

        with open("GUI/probes.json", 'r') as p:
            loaded_probes = json.load(p)

        if probe is not None:
            channel_map = loaded_probes[probe]['channel_map']
            return probe, channel_map
        else:
            print("Probe not found!")