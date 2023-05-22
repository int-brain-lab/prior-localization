import matplotlib.pyplot as plt
import pandas as pd
from ibllib.atlas.flatmaps import plot_swanson
from ibllib.atlas import BrainRegions

from brain_plots_utils import format_results,plot_region_variable_branwide,shiftedColorMap


# import the .parquet file containing decoding results
df_results = pd.read_parquet("results/2022-04-14_decode_pLeft_optBay_Lasso_align_goCue_times_200_pseudosessions_regionWise_timeWindow_-0_6_-0_1_pseudoSessions_mergedProbes.parquet")

# format the results by computing corrected R2 scores per region and 95% quantiles for significativity assessement
df_reg,df_sess_reg,df_reg_pseudo = format_results(df_results,correction='median')

# the plotting function takes a subset of the main dataframe (ROI for regions of interest), for plotting
df_roi = df_reg[ df_reg['R2_test'] > df_reg['pseudo_95_quantile'] ]
plot_region_variable_branwide( df_reg, df_roi, var_name='corrected_R2_test',var_label=r'$R^2$', title=r'prior decoding $R^2$')

br = BrainRegions()

cmap=plt.get_cmap('Blues')
cmap = shiftedColorMap(cmap, start=0, midpoint=0, stop=1.0, name='shiftedcmap')
plt.figure()
ax_swanson = plot_swanson(acronyms=df_roi.index.get_level_values(0).values, values=df_roi['corrected_R2_test'].values,br=br,cmap=cmap)
ax_swanson.set_title(r'Prior decoding $R^2$')
cbar = plt.colorbar(mappable=ax_swanson.images[0])
plt.show()