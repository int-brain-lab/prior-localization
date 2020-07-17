from oneibl import one
import numpy as np
from brainbox.modeling import glm
from export_funs import trialinfo_to_df
from datetime import date
import pickle

one = one.ONE()
kernlen = 0.6
nbases = 10


def fit_session_sk(trialsdf, spk_times, spk_clu,
                   t_before=0.4, t_after=0.6, alpha=0):
    trialsdf['trial_start'] = trialsdf['stimOn_times'] - t_before
    trialsdf['trial_end'] = trialsdf['feedback_times'] + t_after
    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
                'feedbackType': 'value',
                'feedback_times': 'timing',
                'contrastLeft': 'value',
                'contrastRight': 'value',
                'goCue_times': 'timing',
                'stimOn_times': 'timing',
                'trial_start': 'timing',
                'trial_end': 'timing',
                'bias': 'value'}
    nglm = glm.NeuralGLM(trialsdf, spk_times, spk_clu, vartypes)
    cosbases_long = glm.full_rcos(kernlen, nbases, nglm.binf)
    nglm.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
                              cond=lambda tr: np.isfinite(tr.contrastLeft),
                              desc='Kernel conditioned on L stimulus onset')
    nglm.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
                              cond=lambda tr: np.isfinite(tr.contrastRight),
                              desc='Kernel conditioned on R stimulus onset')
    nglm.add_covariate_timing('correct', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == 1,
                              desc='Kernel conditioned on correct feedback')
    nglm.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == -1,
                              desc='Kernel conditioned on incorrect feedback')
    nglm.compile_design_matrix()
    nglm.bin_spike_trains()
    nglm.fit(method='sklearn', alpha=alpha)
    combined_weights = nglm.combine_weights()
    return nglm, combined_weights


if __name__ == "__main__":
    subject = 'ZM_2240'
    sessdate = '2020-01-22'
    ids = one.search(subject=subject, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.clusters'])
    trialsdf = trialinfo_to_df(ids[0], maxlen=2.)
    spk_times = one.load(ids[0], ['spikes.times'])[0]
    spk_clu = one.load(ids[0], ['spikes.clusters'])[0]
    alphas = np.linspace(0, 0.001, 20)
    wtdict = {}
    for alpha in alphas:
        nglm, combined_weights = fit_session_sk(trialsdf.copy(), spk_times, spk_clu, alpha=alpha)
        wtdict[alpha] = (nglm, combined_weights)
    outdict = {'regularizations': wtdict, 'spk_times': spk_times, 'spk_clu': spk_clu,
               'trialsdf': trialsdf}
    currdate = str(date.today())
    fw = open(f'regularization_test_{subject}_sessdate_{sessdate}_fitdate_{currdate}.p', 'wb')
    pickle.dump(outdict, fw)
    fw.close()
