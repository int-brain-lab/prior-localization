import pickle
from behavior_models.models.utils import format_data as format_data_mut
import pandas as pd
import glob
from braindelphi.decoding.settings import *
import models.utils as mut
from braindelphi.params import FIT_PATH
from braindelphi.decoding.settings import modeldispatcher
from tqdm import tqdm

SAVE_KFOLDS = False

date = '60-06-2022'
finished = glob.glob(str(FIT_PATH.joinpath(kwargs['neural_dtype'], "*", "*", "*", "*%s*" % date)))

weight_indexers = ['subject', 'eid', 'probe', 'region'] # 'region'
weightsdict = {}
for fn in tqdm(finished):
    if 'pseudo_id_-1' in fn :
        fo = open(fn, 'rb')
        result = pickle.load(fo)
        fo.close()
        for i_run in range(len(result['fit'])):
            weightsdict = {**weightsdict, **{(tuple( (result[x][0] if isinstance(result[x], list) else result[x]) for x in weight_indexers)
                                            + (result['pseudo_id'],
                                                i_run + 1))
                                            : np.vstack(result['fit'][i_run]['weights'])}}

weights = pd.Series(weightsdict).reset_index()
weights.columns = ['subject','session','hemisphere','region','pseudo_id','run_id','weights']

estimatorstr = strlut[ESTIMATOR]
start_tw, end_tw = TIME_WINDOW
model_str = 'interIndividual' if isinstance(MODEL, str) else modeldispatcher[MODEL]
fn = str(FIT_PATH.joinpath(kwargs['neural_dtype'], '_'.join([date, 'decode', TARGET,
                                                               model_str if TARGET in ['prior',
                                                                                                        'pLeft']
                                                               else 'task',
                                                               estimatorstr, 'align', ALIGN_TIME, str(N_PSEUDO),
                                                               'pseudosessions',
                                                               'regionWise' if SINGLE_REGION else 'allProbes',
                                                               'timeWindow', str(start_tw).replace('.', '_'),
                                                               str(end_tw).replace('.', '_')])))

if ADD_TO_SAVING_PATH != '':
    fn = fn + '_' + ADD_TO_SAVING_PATH

weights_fn =  fn + '.weights.pkl'

with open(weights_fn, 'wb') as f:
    pickle.dump(weights, f)
