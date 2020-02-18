function [cellweights, cellstats] = full_fit(trialfilename)
%FIT_TRIAL Fits GLM to all neurons in a given trial
%   Fits a GLM using neuroGLM to the trial-by-trial data in trialfilename.
disp(strcat('Fitting file:', trialfilename))
mintrials = 8;
wts_per_kern = 10;
trialdata = load(trialfilename);
expt = buildGLM.initExperiment('s', 0.025, trialdata.subject_name, 'brainwide_map');
expt = buildGLM.registerTiming(expt, 'stimOn', 'stimulus on time');
expt = buildGLM.registerTiming(expt, 'feedback_t', 'Time feedback was administered');

cell_ids = trialdata.clusters;
fitobjs = struct;
cellweights = struct;
cellstats = struct;
[~, sessname, ~] = fileparts(trialfilename);
for i = 1:length(cell_ids)
    cellname = strcat('cell', num2str(cell_ids(i)));
    fitobjs.(cellname) = buildGLM.registerSpikeTrain(expt, cellname, 'the neuron');
    goodtrialnum = 1;
    for j = 1:length(trialdata.trials)
        currtrial = trialdata.trials{j};
        if isempty(currtrial.spikes(currtrial.clu == cell_ids(i)))
            continue
        end
        trialobj = buildGLM.newTrial(fitobjs.(cellname), 0.5 + currtrial.feedback_times);
        trialobj.stimOn = currtrial.stimOn_times;
        trialobj.feedback_t = currtrial.feedback_times;
        trialobj.(cellname) = currtrial.spikes(currtrial.clu == cell_ids(i));
        try
            fitobjs.(cellname) = buildGLM.addTrial(fitobjs.(cellname), trialobj, goodtrialnum);
        catch
            disp('SOMETHING IS BROKEN')
            continue
        end
        goodtrialnum = goodtrialnum + 1;
    end
    if (goodtrialnum == 1) || (numel(fitobjs.(cellname).trial) < mintrials)
        clear fitobjs.(cellname)
        continue
    end
    dspec = buildGLM.initDesignSpec(fitobjs.(cellname));
    bs = basisFactory.makeSmoothTemporalBasis('raised cosine', 0.6, wts_per_kern, expt.binfun);
    dspec = buildGLM.addCovariateTiming(dspec, 'stimOn', 'stimOn', 'Stimulus on', bs);
    dspec = buildGLM.addCovariateTiming(dspec, 'feedback_t', 'feedback_t', 'feedback time', bs);
    dm = buildGLM.compileSparseDesignMatrix(dspec, 1:goodtrialnum - 1);
    dm = buildGLM.removeConstantCols(dm);
    dm = buildGLM.addBiasColumn(dm);  % comment this out if using GLMfit
    if nnz(dm.X) / numel(dm.X) > 0.70
        dm.X = full(dm.X);
    end
    disp(strcat('fitting :', cellname));
    y = buildGLM.getBinnedSpikeTrain(fitobjs.(cellname), cellname);
    wInit = dm.X \ y; % least sq for init
    fnlin = @nlfuns.exp; % Inverse link function
    lfunc = @(w)(glms.neglog.poisson(w, dm.X, y, fnlin)); % Loss func
    opts = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'GradObj', 'on', 'Hessian', 'on');
    [wml, ~, ~, ~, ~, hessian] = fminunc(lfunc, wInit, opts);
    wvar = diag(inv(hessian));
    cellweights.(cellname) = buildGLM.combineWeights(dm, wml);
    cellstats.(cellname) = wvar;
end
save(strcat('./fits/', sessname, '_fit.mat'), 'cellweights', 'cellstats');