import os
import shutil
import tempfile
import json
import subprocess

class MyFunctions():
    def __init__(self):
        print("initializing my functions")

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

    def search_probe(self, path):
        if 'P1' in path:
            probe = "ASSY-37-P-1"
        elif 'H4' in path:
            probe = "ASSY-37-H4"
        elif 'H6b' in path:
            probe = "ASSY-37-H6b"
        elif 'E1' in path:
            probe = "ASSY-1-E-1"
        else:
            probe = None

        with open("GUI/probes.json", 'r') as p:
            loaded_probes = json.load(p)

        if probe is not None:
            channel_map = loaded_probes[probe]['channel_map']
            return probe, channel_map
        else:
            print("Probe not found!")

    # def button_clicked(self, base_dir, name):
    #     print(f'Button {name} clicked')
    #     print(f'Folder Path: {os.path.join(base_dir, name, 'unfiltered')}')
    #     recording_path = os.path.join(base_dir, name, 'unfiltered')
    #     chan_map = [
    #         29, 19, 18, 28, 30, 20, 17, 21,
    #         31, 22, 16, 23, 27, 26, 25, 24,
    #         7, 6, 5, 4, 8, 10, 9, 3,
    #         11, 2, 12, 1, 13, 0, 14, 15
    #     ]
    #     probe = "ASSY-37-P-1"
    #
    #     cmd = [
    #         "python",
    #         "GUI/sort_data.py",
    #         "--data", recording_path,
    #         "--probe", probe,
    #         "--chanmap", json.dumps(chan_map),
    #         "--sorter", "kilosort4"
    #     ]
    #     p = subprocess.Popen(cmd)
