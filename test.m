trialfilename = './data/bias_0.2_LeftTrial_Nonzerocontrast.mat';
kernlen = 0.6;
wts_per_kern = 10;
binw = 0.02;
disp(strcat('Fitting file:', trialfilename))
mintrials = 5;

trialdata = load(trialfilename);
expt = buildGLM.initExperiment('s', binw, trialdata.subject_name, 'brainwide_map');
expt = buildGLM.registerTiming(expt, 'stimOn', 'stimulus on time');
expt = buildGLM.registerTiming(expt, 'feedback_t', 'Time feedback was administered');
expt = buildGLM.registerValue(expt, 'prior', 'Prior estimate');

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
        trialobj = buildGLM.newTrial(fitobjs.(cellname), kernlen + currtrial.feedback_times);
        trialobj.stimOn = currtrial.stimOn_times;
        trialobj.feedback_t = currtrial.feedback_times;
        trialobj.prior = currtrial.prior;
        trialobj.(cellname) = currtrial.spikes(currtrial.clu == cell_ids(i));
        try
            fitobjs.(cellname) = buildGLM.addTrial(fitobjs.(cellname), trialobj, goodtrialnum);
        catch
            disp('SOMETHING IS BROKEN')
            quit(1)
        end
        goodtrialnum = goodtrialnum + 1;
    end
    if (goodtrialnum == 1) || (numel(fitobjs.(cellname).trial) < mintrials)
        clear fitobjs.(cellname)
        continue
    end
    dspec = buildGLM.initDesignSpec(fitobjs.(cellname));
    binfun = expt.binfun;
    bs = basisFactory.makeSmoothTemporalBasis('raised cosine', kernlen, wts_per_kern, binfun);
    dspec = buildGLM.addCovariateTiming(dspec, 'stimOn', 'stimOn', 'Stimulus on', bs);
    dspec = buildGLM.addCovariateTiming(dspec, 'feedback_t', 'feedback_t', 'feedback time', bs);
    % Make boxcar associated with the value of the prior estimate for that trial
    bs2 = basisFactory.makeSmoothTemporalBasis('boxcar', kernlen, 1, binfun);
    stimHandle = @(trials, expt) trials.prior * basisFactory.boxcarStim(binfun(trials.stimOn), ...
        binfun(trials.feedback_t), binfun(trials.feedback_t + kernlen));
    dspec = buildGLM.addCovariate(dspec, 'prior', 'prior effect on activity', stimHandle, bs2);
    dm = buildGLM.compileSparseDesignMatrix(dspec, 1:goodtrialnum - 1);
    dm = buildGLM.removeConstantCols(dm);
    dm = buildGLM.addBiasColumn(dm);  % comment this out if using GLMfit
    if nnz(dm.X) / numel(dm.X) > 0.20
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
    cellstats.(cellname) = sqrt(wvar);
end
% save(strcat('./fits/', sessname, '_fit.mat'), 'cellweights', 'cellstats');