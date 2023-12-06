%% Q3 Part one: segmentation using classification
%%

PCGCellArray = train_recordings;
annotationsArray = train_annotations;
numberOfStates = 4;
numPCGs = length(PCGCellArray);
Fs = 1000;
% A matrix of the values from each state in each of the PCG recordings:
state_observation_values = cell(numPCGs,numberOfStates);
%state_observation_label = cell(numPCGs,1);
PCG_Features_all = [];
PCG_states_all = [];

for PCGi = 1:length(PCGCellArray)
    
    PCG_audio = PCGCellArray{PCGi};
    
    S1_locations = annotationsArray{PCGi,1};
    S2_locations = annotationsArray{PCGi,2};
    
    [PCG_Features, featuresFs] = getSpringerPCGFeatures(PCG_audio, Fs);
    
    PCG_states = labelPCGStates(PCG_Features(:,1),S1_locations, S2_locations, featuresFs);
    % your code here:
    PCG_Features_all = [PCG_Features_all ; PCG_Features];
    PCG_states_all = [PCG_states_all ; PCG_states];
end
    
save('PCG_Features_all.mat','PCG_Features_all')
save('PCG_states_all.mat','PCG_states_all')    
%% part 4-d
load('test_recording.mat');
load('test_annotations.mat');
PCGCellArray = test_recordings;
annotationsArray = test_annotations;
numberOfStates = 4;
numPCGs = length(PCGCellArray);
Fs = 1000;
% A matrix of the values from each state in each of the PCG recordings:
state_observation_values = cell(numPCGs,numberOfStates);

for PCGi = 1:length(PCGCellArray)
    PCG_audio = PCGCellArray{PCGi};
    
    S1_locations = annotationsArray{PCGi,1};
    S2_locations = annotationsArray{PCGi,2};
    
    [PCG_Features_test, featuresFs] = getSpringerPCGFeatures(PCG_audio, Fs);
    
    PCG_states_test = labelPCGStates(PCG_Features_test(:,1),S1_locations, S2_locations, featuresFs);
end

save('PCG_Features_test.mat','PCG_Features_test')
save('PCG_states_test.mat','PCG_states_test')    