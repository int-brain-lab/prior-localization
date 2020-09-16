kernlen = 0.6;
nbases = 10;

load('synthspikes.mat');
iblglm_add2path;
expt = buildGLM.initExperiment('s', 0.02, 'Robomouse', 'testing');
expt = buildGLM.registerTiming(expt, 'stimOn', 'stimulus on time');
currfit = buildGLM.registerSpikeTrain(expt, 'cell1', 'neuron spiking');
for j = 1:length(stimt) - 1
    trialobj = buildGLM.newTrial(currfit, fdbkt(j) + kernlen);
    trialspikes = spk_times(j);
    trialobj.stimOn = stimt(j);
    if isempty(trialspikes{1})
        trialobj.cell1 = [0.0001, 0.0002];
    else
        trialobj.cell1 = trialspikes{1};
    end
    currfit = buildGLM.addTrial(currfit, trialobj, j);
end

dspec = buildGLM.initDesignSpec(currfit);
binfun = expt.binfun;
bs = basisFactory.makeSmoothTemporalBasis('raised cosine', kernlen, nbases, binfun);
dspec = buildGLM.addCovariateTiming(dspec, 'stim', 'stimOn', 'Stimulus onset kernel', bs, 0);
dm = buildGLM.compileSparseDesignMatrix(dspec, 1:length(stimt) - 1);
dm = buildGLM.removeConstantCols(dm);
dm = buildGLM.addBiasColumn(dm);
dm.X = full(dm.X);
y = buildGLM.getBinnedSpikeTrain(currfit, 'cell1');
wInit = dm.X \ y;
fnlin = @nlfuns.exp;
lfunc = @(w)(glms.neglog.poisson(w, dm.X, y, fnlin));
opts = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'GradObj', 'on', 'Hessian', 'on');
[wml, ~, ~, ~, ~, hessian] = fminunc(lfunc, wInit, opts);
wvar = diag(inv(hessian));
weights = buildGLM.combineWeights(dm, wml);
stats = buildGLM.combineWeights(dm, wvar);
intercepts = wml(1);
savedm = dm.X;
save('synthmatfit.mat', 'weights', 'stats', 'savedm', 'y', 'intercepts');
exit;