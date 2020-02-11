trialdata = load('/home/berk/Documents/Projects/ibl_shared/DY_010_recording_data.mat');

duration = 1.;
preStimTime = 0.1;
numcells = 1;
cellname = 'cell195';

expt = buildGLM.initExperiment('s', 0.001, 'IBL_DY_006_BWM', 'contrast');
expt = buildGLM.registerTiming(expt, 'stimOn', 'Stimulus on time');
expt = buildGLM.registerValue(expt, 'contrast', 'Contrast of stimulus');

cells = struct;
if numcells > 1
    for i = 0:numcells - 1
        cellname = strcat('cell', num2str(i));
        expt = buildGLM.registerSpikeTrain(expt, cellname, 'a neuron');
        cells.(cellname) = trialdata.(cellname);
    end
elseif numcells == 1
    expt = buildGLM.registerSpikeTrain(expt, cellname, 'our neuron');
    cells.(cellname) = trialdata.(cellname);
end

stimOnTimes = trialdata.stimOnTimes;
contrast = trialdata.contrast;
goodtrials = 0;
badtrials = [];
for i = 1:length(stimOnTimes)
    starttime = stimOnTimes(i) - preStimTime;
    endtime = starttime + (duration - preStimTime);
    currtrial = buildGLM.newTrial(expt, duration);
    currtrial.stimOn = stimOnTimes(i) - starttime;
    currtrial.contrast = contrast(i);
    cellnames = fieldnames(cells);
    for k = 1:numel(cellnames)
        cellspikes = cells.(cellnames{k});
        trialinds = starttime <= cellspikes & cellspikes <= endtime;
        trialspikes = cellspikes(trialinds);
        currtrial.(cellnames{k}) = trialspikes - starttime;
    end
    if sum(trialinds) == 0
        badtrials = [badtrials i];
        continue
    end
    goodtrials = goodtrials + 1;
    expt = buildGLM.addTrial(expt, currtrial, goodtrials);
end

dspec = buildGLM.initDesignSpec(expt);
bs = basisFactory.makeSmoothTemporalBasis('boxcar', 0.6, 50, expt.binfun);
dspec = buildGLM.addCovariateTiming(dspec, 'stimOn', 'stimOn', 'stimulus on', bs);
dm = buildGLM.compileSparseDesignMatrix(dspec, 1:goodtrials);
dm.X = full(dm.X);

y = buildGLM.getBinnedSpikeTrain(expt, cellname);

[w, dev, stats] = glmfit(dm.X, y, 'poisson', 'link', 'log');
ws = buildGLM.combineWeights(dm, w(1:end-1));
plot(ws.stimOn.data)
        