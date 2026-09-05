clc; clear all; close all;
% --- Radar Parameters ---
Nr = 60; % Number of range bins (height)
Nc = 100; % Number of chirps (width)

A = 1.0; % Amplitude

center_range = 25; % Center in range 
center_chirp = 100; % Center in chirps 

B = 200e6; % Bandwidth in Hz..........covenient unit
c = 3e8; % Speed of light (m/s)
range_resolution = c / (2 * B); % Range resolution

spread_range = 12; % size of target
spread_chirp = 12; % size of target

% --- Calculate Range Bin Size ---
range_bin_size = range_resolution;
disp(['Range Resolution (Range Bin Size): ', num2str(range_bin_size), ' meters']);

% --- Noise Parameters --
% -
noise_level = 0.10;

% --- Create Meshgrid ---
[range_bins, chirps] = meshgrid(1:Nr, 1:Nc); 
range_bins = range_bins'; % Transpose to have range bins along rows
chirps = chirps';       % Transpose to have chirps along columns

% --- Generate Gaussian Target ---
tank_signal = A .* exp(-((chirps - center_chirp).^2 / (2 * spread_chirp.^2)) - ((range_bins - center_range).^2 / (2 * spread_range.^2)));

% --- Add Noise ---
noise = noise_level * randn(Nr, Nc); % Noise should have the same dimensions as tank_signal
range_time_map = tank_signal + noise;
% --- Thresholding ---

threshold_factor = 3;
threshold = mean(range_time_map(:)) + threshold_factor * std(range_time_map(:));
detected_tank = range_time_map > threshold;
se = strel('disk', 3);
detected_tank_cleaned = imopen(detected_tank, se);
detected_tank_cleaned = imclose(detected_tank_cleaned, se);
stats = regionprops(detected_tank_cleaned, 'Centroid');
if ~isempty(stats)
    detected_center_chirp = round(stats(1).Centroid(1)); % Centroid is (column, row) = (Chirp, Range Bin)
    detected_center_range = round(stats(1).Centroid(2));
    disp(['Tank detected near Range Bin: ', num2str(detected_center_range), ', Chirp: ', num2str(detected_center_chirp)]);
else
    disp('No tank detected based on the threshold.');
end
% --- Plotting ---
figure('Position', [100, 100, 800, 400]);
subplot(1, 2, 1);
imagesc(1:Nc, 1:Nr, range_time_map, [-0.2, A]); % Chirps on x, Range Bins on y
colormap(flipud(brewermap(256, 'RdBu'))); % Use 'jet' if brewermap is unavailable
colorbar;
caxis([-0.25, 1.0]); % Adjust color limits to match the scale
xlabel('Chirps');
ylabel('Range Bin');
title('Tank');
set(gca, 'YDir', 'normal');
hold on;
if ~isempty(stats)
    plot(detected_center_chirp, detected_center_range, 'w+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
subplot(1, 2, 2);
imagesc(1:Nc, 1:Nr, detected_tank_cleaned); % Chirps on x, Range Bins on y
colormap(gray(2));
xlabel('Chirps');
ylabel('Range Bin');
title('Detected Tank');
set(gca, 'YDir', 'normal');
