The files in this directory will allow you to convert the decoding pipeline output into a form that can be
plotted in the figures of the BWM paper.
The processing done in this step is primarily to read in the output summary tables of the decoding
pipeline and combine regions.  Specifically, the decoding pipeline outputs are sorted according to region
and session, but sessions of the same region need to be combined using fisher's method to plot per-region
results in the paper.

directories and saving information...

To process block summary tables, run the file
`
plot_decoding_block_panel.py
`
up to the end of the first block (spyder) denoted by `# %%`
