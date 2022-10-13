import sys, os
import json
import numpy as np
sys.path.append(os.getcwd())
from os.path import join, isfile
import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)

with open('action_label.json', 'r') as f:
    action_json = json.load(f)

sys.path.append(paths['CODE'])
BEHAVE_PATH = paths['BEHAVE_PATH']

def process_action():
    outdir = paths['PROCESSED_PATH']

    for action_label in action_json['action_label']:
        outfolder = join(outdir, action_label['name'], action_label['frame'])
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfile = join(outfolder, 'action.npz')
        data = np.array(int(action_label['label']))
        np.savez(outfile, action=data)


    outdir = BEHAVE_PATH
    for action_label in action_json['action_label']:
        outfolder = join(outdir, action_label['name'], action_label['frame'])
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfile = join(outfolder, 'action.npz')
        data = np.array(int(action_label['label']))
        np.savez(outfile, action=data)
        
if __name__ == "__main__":
    process_action()