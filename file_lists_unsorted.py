##this is a file to keep track of all the data files 
##that contain the unsorted neuron hash.  Note that 
##some of the early recordings (days 1-5) don't
##have unsorted timestamps saved, so they aren't included here.

import os
from sys import platform as _platform
import h5py

if _platform == 'win32':
	root = r'K:/'
elif _platform == 'darwin':
	root = "/Volumes/Untitled"

##save location
save_loc = os.path.join(root,"Ryan/DS_animals/results")

animals = ['S1','S2','S3']

behavior_files = [
##S1 files
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D01.hdf5"), ##first 6 days have only 1 lever
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D02.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D03.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D04.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D05.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D06.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D07.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D08.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D09.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D13.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D17.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D18.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_D19.hdf5"),
##S1 files with ephys recordings
# os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R01.hdf5"),
# # os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R02.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R03.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R04.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R05.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R06.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R07.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R08.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R09.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R13.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R17.hdf5"),
##S2 files
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D01.hdf5"), ##first 6 days only have 1 lever
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D02.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D03.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D04.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D05.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D06.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D07.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D08.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D09.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D13.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D17.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D18.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_D19.hdf5"),
##S2 files with ephys recordings
# os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R01.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R02.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R03.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R04.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R05.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R06.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R07.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R08.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R09.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R13.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R17.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R18.hdf5"),
##S3 files
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D01.hdf5"), ##first 6 days have only 1 lever
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D02.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D03.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D04.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D05.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D06.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D07.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D08.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D09.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D17.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_D18.hdf5"),
##S3 files with ephys recordings
# os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R01.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R02.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R03.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R04.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R05.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R06.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R07.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R08.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R09.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R17.hdf5")
]

ephys_files = [
##S1 files
# os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R01_h.hdf5"), ##these didn't have unsorted timestamps saved
# # os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R02r.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R03_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R04_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R05_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R06_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R07_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R08_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R09_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R10_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R11_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R12_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R13_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R14_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R15_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R16_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R17_h.hdf5"),
##S2 files
# #os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R01r.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R02_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R03_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R04_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R05_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R06_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R07_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R08_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R09_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R10_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R11_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R12_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R13_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R14_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R15_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R16_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R17_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R18_h.hdf5"),
##S3 files
# os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R01_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R02_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R03_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R04_h.hdf5"),
# os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R05_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R06_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R07_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R08_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R09_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R10_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R11_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R12_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R14_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R15_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R16_h.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R17_h.hdf5"),
]

##run a test on loading that all files exist and can be opened
##also make sure all files are properly matched
e_behavior = [x for x in behavior_files if x[-8]=='R']

for f_behavior,f_ephys in zip(e_behavior,ephys_files):
	behavior_id = f_behavior[-7:-5]
	ephys_id = f_ephys[-9:-7]
	assert behavior_id == ephys_id
	try:
		f = h5py.File(f_behavior,'r')
		f.close()
		f = h5py.File(f_ephys,'r')
		f.close()
	except OSError:
		print("Warning: data file {} not detected".format(behavior_id))

### a function to get a dictionary of file paths split by animal.
def split_behavior_by_animal(match_ephys=False):
	global animals
	global behavior_files
	global e_behavior
	by_animal = {}
	if match_ephys:
		all_files = e_behavior
	else:
		all_files = behavior_files
	for a in animals:
		files = [x for x in all_files if x[-11:-9]==a]
		by_animal[a] = files
	return by_animal
### a function to get a dictionary of file paths split by animal.
def split_ephys_by_animal():
	global animals
	global ephys_files
	by_animal = {}
	for a in animals:
		files = [x for x in ephys_files if x[-13:-11]==a]
		by_animal[a] = files
	return by_animal