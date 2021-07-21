import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

basepath = '/home/berk/Documents/'

masterscores = pd.read_parquet(basepath + 'stim_GLM_v_dec.parquet')
masterscores.drop(columns=masterscores.columns[209:], inplace=True)
masterscores = masterscores.reset_index().set_index(['eid', 'probe', 'region', 'clu_id'])

prior_master = pd.read_parquet(basepath + 'decoding_encoding_comparison_2021-07-02.parquet')
prior_master.drop(index=['void', 'root'], level='region', inplace=True)
prior_master.drop(columns=['subject', 'r', 'r_null', 'p_value_r'], inplace=True)
prior_master = prior_master.reset_index().set_index(['eid', 'probe', 'region', 'clu_id'])

dcpath = ('/home/berk/Downloads/ToBerk_noCV_alpha0p03_stimside'
          '_pseudo_kfold_aligned-behavior_all_cells_beryl-atlas_0-100.p')
pdcpath = ('/home/berk/Downloads/prior-prevaction_other-trials_'
           'kfold_aligned-behavior_all_cells_beryl-atlas_600--100.p')
decoding = np.load(dcpath, allow_pickle=True)
pr_dec = np.load(pdcpath, allow_pickle=True)

dec_weights = pd.DataFrame([{'dec_w': w, 'dec_w_transform': x['reg_coefs_encoding'][j],
                             'clu_id': x['cluster_ids'][j], 'subject': x['subject'],
                             'eid': x['eid'], 'probe': x['probe'], 'region': x['region'],
                             'r': x['r'], 'r_null': x['r_null'], 'p_value_r': x['p_value_r']}
                            for i, x in decoding.iterrows() for j, w in enumerate(x['reg_coefs'])])
dec_weights.set_index(['eid', 'probe', 'region', 'clu_id'], inplace=True)

pdec_weights = pd.DataFrame([{'dec_w': w,
                              'clu_id': x['cluster_ids'][j], 'subject': x['subject'],
                              'eid': x['eid'], 'probe': x['probe'], 'region': x['region'],
                              'r': x['r'], 'r_null': x['r_null'], 'p_value_r': x['p_value_r']}
                             for i, x in pr_dec.iterrows() for j, w in enumerate(x['reg_coefs'])])
pdec_weights.set_index(['eid', 'probe', 'region', 'clu_id'], inplace=True)


joindf = masterscores.join(dec_weights, how='inner')
joindf['mean_w'] = joindf.dec_w.apply(np.mean)
joindf['mean_enc_w'] = joindf.dec_w_transform.apply(np.mean)
joindf['std_w'] = joindf.dec_w.apply(np.std)
joindf['std_enc_w'] = joindf.dec_w_transform.apply(np.mean)
joindf['abs_w'] = joindf.mean_w.apply(np.abs)
joindf['abs_enc_w'] = joindf.mean_enc_w.apply(np.abs)
joindf['zsc'] = zscore(joindf.loc[:, 'baseline':'run199'].values, axis=1)[:, 0]

pr_jdf = prior_master.join(pdec_weights, how='inner')
pr_jdf['mean_w'] = pr_jdf.dec_w.apply(np.mean)
pr_jdf['std_w'] = pr_jdf.dec_w.apply(np.std)
pr_jdf['abs_w'] = pr_jdf.mean_w.apply(np.abs)
pr_jdf['zsc'] = zscore(pr_jdf.loc[:, 'baseline':'run199'].values, axis=1)[:, 0]


# Compare pure decoding weights with the z-scored performance
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
ax[3].set_visible(False)
reg = LinearRegression().fit(joindf['zsc'], joindf['abs_w'])
reg_enc = OLS(joindf['zsc'], joindf['abs_enc_w']).fit().rsquared
reg_prior = OLS(pr_jdf['zsc'], pr_jdf['abs_w']).fit().rsquared

sns.regplot(data=joindf, x='zsc', y='abs_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq}'}, ax=ax[0])
sns.regplot(data=joindf, x='zsc', y='abs_enc_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq_enc}'}, ax=ax[1])
sns.regplot(data=pr_jdf, x='zsc', y='abs_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq_prior}'}, ax=ax[2])

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
ax[0].set_ylabel('Decoder weight')
ax[1].set_ylabel('Decoder weight (cov transformed)')
ax[2].set_ylabel('Decoder weight')
ax[0].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[1].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[2].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[0].set_title('Decoder weights vs z-scored GLM scores')
ax[1].set_title('Transformed decoder weights vs z-scored GLM scores')
ax[2].set_title('Prior decoder weights vs z-scored GLM scores')


# Compare pure decoding weights with the z-scored performance only for sig. decoding region recs
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
ax[3].set_visible(False)
mask = joindf['p_value_r'] >= 0.95
prmask = pr_jdf['p_value_r'] >= 0.95
rsq = OLS(joindf['abs_w'][mask],
          np.vstack([joindf['zsc'][mask], np.ones_like(joindf['zsc'][mask])]).T).fit().rsquared
rsq_enc = OLS(joindf['abs_enc_w'][mask], joindf['zsc'][mask]).fit().rsquared
rsq_prior = OLS(pr_jdf['abs_w'][prmask], pr_jdf['zsc'][prmask]).fit().rsquared

sns.regplot(data=joindf[mask], x='zsc', y='abs_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq}'}, ax=ax[0])
sns.regplot(data=joindf[mask], x='zsc', y='abs_enc_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq_enc}'}, ax=ax[1])
sns.regplot(data=pr_jdf[prmask], x='zsc', y='abs_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq_prior}'}, ax=ax[2])

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
ax[0].set_ylabel('Decoder weight')
ax[1].set_ylabel('Decoder weight (cov transformed)')
ax[2].set_ylabel('Decoder weight')
ax[0].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[1].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[2].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[0].set_title('Decoder weights vs z-scored GLM scores\nOnly significant decoding')
ax[1].set_title('Transformed decoder weights vs z-scored GLM scores\nOnly significant decoding')
ax[2].set_title('Prior decoder weights vs z-scored GLM scores\nOnly significant decoding')
plt.savefig(basepath + 'zsc_glm_score_v_dec_w.png')

# Compare pure decoding weights with the z-scored performance only for doubly significant neurons
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
ax[3].set_visible(False)
mask = (joindf['p_value_r'] >= 0.95) & (joindf['zsc'] >= 1.63)
prmask = (pr_jdf['p_value_r'] >= 0.95) & (pr_jdf['zsc'] >= 1.63)
rsq = OLS(joindf['zsc'][mask], joindf['abs_w'][mask]).fit().rsquared

rsq_enc = OLS(joindf['zsc'][mask], joindf['abs_enc_w'][mask]).fit().rsquared
rsq_prior = OLS(pr_jdf['zsc'][prmask], pr_jdf['abs_w'][prmask]).fit().rsquared

sns.regplot(data=joindf[mask], x='zsc', y='abs_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq}'}, ax=ax[0])
sns.regplot(data=joindf[mask], x='zsc', y='abs_enc_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq_enc}'}, ax=ax[1])
sns.regplot(data=pr_jdf[prmask], x='zsc', y='abs_w', scatter_kws={'alpha': 0.2},
            line_kws={'color': 'orange', 'label': f'Rsq = {rsq_prior}'}, ax=ax[2])

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
ax[0].set_ylabel('Decoder weight')
ax[1].set_ylabel('Decoder weight (cov transformed)')
ax[2].set_ylabel('Decoder weight')
ax[0].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[1].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[2].set_xlabel(r'Z-Scored GLM $R^2$ relative to null')
ax[0].set_title('Decoder weights vs z-scored GLM scores\nOnly significant GLM/dec')
ax[1].set_title('Transformed decoder weights vs z-scored GLM scores\nOnly significant GLM/dec')
ax[2].set_title('Prior decoder weights vs z-scored GLM scores\nOnly significant GLM/dec')