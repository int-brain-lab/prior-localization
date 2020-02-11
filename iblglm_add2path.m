%IBLGLM_ADD2PATH Temporarily add repo full path to Matlab path.
%
%   Call this Matlab script at the beginning of a Matlab session, before 
%   running any file in the repository.
%
%   It is *not* recommended to permanently add the 'neuroGLM' folder and sub-
%   folders of this repository to the Matlab path. Since Matlab does not 
%   have any way to deal with function duplicates, it might cause clashes 
%   with other projects.

repo_path = fileparts(mfilename('fullpath'));
addpath(genpath([repo_path filesep 'data']));

glm_path = [repo_path filesep 'neuroGLM'];
addpath(glm_path);

clear matlab_path folder_path;