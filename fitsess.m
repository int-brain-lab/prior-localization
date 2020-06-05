function [cellweights, cellstats] = fitsess(trialfilename, wts_per_kern, binw, kernlen)
    %FITSESS Fits GLM to all neurons in a given session, using contrast, stimulus side, and prior
    %    estimate as inputs to the model.
    
    mintrials = 100;
    
    trialdata = load(trialfilename);
    expt = buildGLM.initExperiment('s', binw, trialdata.subject_name, 'brainwide_map');
    expt = buildGLM.registerTiming(expt, 'stimOn', 'stimulus on time');
    expt = buildGLM.registerTiming(expt, 'feedback_t', 'time at which feedback was given');
    expt = buildGLM.registerTiming(expt, 'decstart', 'beginning of window where to fit decision');
    expt = buildGLM.registerValue(expt, 'prior', 'Estimate of the prior (or true block prob)');
    expt = buildGLM.registerValue(expt, 'side', 'Which side the stimulus was on (L = -1, R = 1)');
    expt = buildGLM.registerValue(expt, 'reward', 'Reward (1) or negative fdbck (-1) on the trial');
    expt = buildGLM.registerValue(expt, 'contr', 'Stimulus contrast');
    expt = buildGLM.registerContinuous(expt, 'prvec', 'Prior estimate until feedback');
    expt = buildGLM.registerContinuous(expt, 'whlV', 'Wheel velocity');
    
    
    cell_ids = trialdata.clusters;
    weights = cell(1, length(cell_ids));
    stats = cell(1, length(cell_ids));
    names = cell(1, length(cell_ids));
    [~, sessname, ~] = fileparts(trialfilename);
    
    parfor i = 1:length(cell_ids)
        cellname = strcat('cell', num2str(cell_ids(i)));
        currfit = buildGLM.registerSpikeTrain(expt, cellname, 'neurons spikes');
        goodtrialnum = 1;
        for j = 1:length(trialdata.trials)-1
            currtrial = trialdata.trials{j};
            if isempty(currtrial.spikes(currtrial.clu == cell_ids(i)))
                nospikes = true;
            else
                nospikes = false;
            end
            trialobj = buildGLM.newTrial(currfit, kernlen + currtrial.feedback_times);
            trialobj.stimOn = currtrial.stimOn_times;
            trialobj.feedback_t = currtrial.feedback_times;
            trialobj.decstart = currtrial.feedback_times - (0.5 * kernlen);
            trialobj.prior = currtrial.prior;
            trialobj.reward = currtrial.feedbackType;
            trialobj.whlV = currtrial.velocity';
            tot_len = expt.binfun(trialobj.duration);
            prior_len = expt.binfun(trialobj.feedback_t);
            nextrial = trialdata.trials{j+1};
            currvec = currtrial.prior * ones(prior_len, 1);
            nexvec = nextrial.prior * ones(tot_len - prior_len, 1);
            trialobj.prvec = [currvec; nexvec];
            if nospikes
                trialobj.(cellname) = [0.00001, 0.0002];
            else
                trialobj.(cellname) = currtrial.spikes(currtrial.clu == cell_ids(i));
            end
            if find(isfinite([currtrial.contrastLeft, currtrial.contrastRight])) == 2
                trialobj.side = 1;
                trialobj.contr = currtrial.contrastRight;
            else
                trialobj.side = -1;
                trialobj.contr = currtrial.contrastLeft;
            end
            currfit = buildGLM.addTrial(currfit, trialobj, goodtrialnum);

            goodtrialnum = goodtrialnum + 1;
        end
        if (goodtrialnum == 1) || (numel(currfit.trial) < mintrials)
            disp(strcat(cellname, ' failed with ', num2str(goodtrialnum), 'good trials'));
            continue
        end
        dspec = buildGLM.initDesignSpec(currfit);
        binfun = expt.binfun;
        bs = basisFactory.makeSmoothTemporalBasis('raised cosine', kernlen, wts_per_kern, binfun);
        % decbs = basisFactory.makeSmoothTemporalBasis('raised cosine', kernlen * 0.5, wts_per_kern, binfun);
        stonHandle = @(trial, expt) (trial.contr * basisFactory.deltaStim(binfun(trial.stimOn), binfun(trial.duration)));
        stonL = @(trial) (trial.side == -1);
        stonR = @(trial) (trial.side == 1);
        correct = @(trial) (trial.reward == 1);
        incorr = @(trial) (trial.reward == -1);
        dspec = buildGLM.addCovariate(dspec, 'stonL', 'Stimulus on L modulated by contrast', stonHandle, bs, 0, stonL);
        dspec = buildGLM.addCovariate(dspec, 'stonR', 'Stimulus on R modulated by contrast', stonHandle, bs, 0, stonR);
        dspec = buildGLM.addCovariateTiming(dspec, 'fdbckCorr', 'feedback_t', 'Response to correct feedback',...
            bs, 0, correct);
        dspec = buildGLM.addCovariateTiming(dspec, 'fdbckInc', 'feedback_t', 'Response to incorr fdbck', ...
            bs, 0, incorr);
        % dspec = buildGLM.addCovariateTiming(dspec, 'decL', 'decstart', 'Response to incorr fdbck', ...
        %     decbs, 0, stonL);
        % dspec = buildGLM.addCovariateTiming(dspec, 'decR', 'decstart', 'Response to incorr fdbck', ...
        %     decbs, 0, stonR);
        % dspec = buildGLM.addCovariateRaw(dspec, 'prvec', 'Vector of prior values until fdbck');
        % dspec = buildGLM.addCovariateRaw(dspec, 'whlV', 'Gain on wheel velocity');
        dm = buildGLM.compileSparseDesignMatrix(dspec, 1:goodtrialnum - 1);
        dm = buildGLM.removeConstantCols(dm);
        dm = buildGLM.addBiasColumn(dm);
        if nnz(dm.X) / numel(dm.X) > 0.20
            dm.X = full(dm.X);
        end
        disp(strcat('fitting :', cellname))
        y = buildGLM.getBinnedSpikeTrain(currfit, cellname);
        wInit = dm.X \ y;
        fnlin = @nlfuns.exp;
        lfunc = @(w)(glms.neglog.poisson(w, dm.X, y, fnlin));
        opts = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'GradObj', 'on', 'Hessian', 'on');
        [wml, ~, ~, ~, ~, hessian] = fminunc(lfunc, wInit, opts);
        wvar = diag(inv(hessian));
        % [wml, ~, wstat] = glmfit(dm.X, y, 'poisson', 'link', 'log', 'constant', 'off');
        % wvar = wstat.se.^2;
        weights{i} = buildGLM.combineWeights(dm, wml);
        stats{i} = buildGLM.combineWeights(dm, wvar);
        names{i} = cellname;
    end
    
    cellweights = struct;
    cellstats = struct;
    for k = 1:length(names)
        if ~ischar(names{k})
            continue
        else
            cellweights.(names{k}) = weights{k};
            cellstats.(names{k}) = stats{k};
        end
    end
save(strcat('./fits/', sessname, '_fit.mat'), 'cellweights', 'cellstats');
    