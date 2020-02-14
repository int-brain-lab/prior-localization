load('./data/DY_010_2020-02-04.mat');

duration = 2.;
preStimTime = 0.6;
postStimTime = duration - preStimTime;
numcells = length(cluster_ids);

expt = buildGLM.initExperiment('s', 0.001, 'IBL_DY_010_BWM', 'contrast');
expt = buildGLM.registerTiming(expt, 'stimOn', 'Stimulus on time');
expt = buildGLM.registerValue(expt, 'contrast', 'Contrast of stimulus');

goodcellnames = fieldnames(spiket);
goodcells = struct;
fitobs = struct;
for i = 1:length(cluster_ids)
    cellname = goodcellnames{i};
    trialspkcount = zeros(length(stimtContra), 1);
    for j = 1:length(stimtContra)
        currwind = [stimtContra(j) - preStimTime, stimtContra(j) + postStimTime];
        windspikes = (spiket.(cellname) >= currwind(1)) & (spiket.(cellname) <= currwind(2));
        trialspkcount(j) = sum(windspikes);
    end
    if all(trialspkcount)
        fitobs.(cellname) = buildGLM.registerSpikeTrain(expt, cellname, 'a neuron');
        goodcells.(cellname) = spiket.(cellname);
    end  
end
disp(['Found ',num2str(length(fields(goodcells))),' cells with spikes during every trial']);

goodcellnames = fieldnames(goodcells);
for i = 1:length(stimtContra)
    starttime = stimtContra(i) - preStimTime;
    endtime = starttime + duration;
    currtrial = buildGLM.newTrial(expt, duration);
    currtrial.stimOn = stimtContra(i) - starttime;
    currtrial.contrast = contrastContra(i);
    for k = 1:numel(goodcellnames)
        cellname = goodcellnames{k};
        cellspikes = goodcells.(cellname);
        trialinds = (starttime <= cellspikes) & (cellspikes <= endtime);
        trialspikes = cellspikes(trialinds);
        currtrial.(goodcellnames{k}) = trialspikes - starttime;
        fitobs.(cellname) = buildGLM.addTrial(fitobs.(cellname), currtrial, i);
    end
end


numtrials = numel(stimtContra);
cellweights = struct;
cellstats = struct;
tic
for i = 1:numel(goodcellnames)
    cellname = goodcellnames{i};
    dspec = buildGLM.initDesignSpec(fitobs.(cellname));
    bs = basisFactory.makeSmoothTemporalBasis('raised cosine', 0.8, 25, expt.binfun);
    dspec = buildGLM.addCovariateTiming(dspec, 'stimOn', 'stimOn', 'stimulus on', bs);
    dspec = buildGLM.addCovariateSpiketrain(dspec, 'hist', cellname, 'History filter');
    dm = buildGLM.compileSparseDesignMatrix(dspec, 1:numtrials);
    if nnz(dm.X) / numel(dm.X) > 0.15
        dm.X = full(dm.X);
    end
    disp(strcat('fitting cell', ' ', cellname));
    y = buildGLM.getBinnedSpikeTrain(fitobs.(cellname), cellname);
    [w, dev, stats] = glmfit(dm.X, y, 'poisson', 'link', 'log');
    cellweights.(cellname) = buildGLM.combineWeights(dm, w(2:end));
    cellstats.(cellname) = stats;
end
toc
