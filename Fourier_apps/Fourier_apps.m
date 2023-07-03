%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Applications of the Fourier transform
%      VIDEO: Rhythmicity in walking (gait)
% Instructor: mikexcohen.com
%
%%


% Load the data. If you get an error on this line,
% make sure MATLAB is in the same directory as the data.
% Or you can add the folder where the data are kept to the MATLAB path.
load gait.mat




figure(1), clf

% plot the gait speed for Parkinson's patient and control
subplot(311)
plot(park(:,1),park(:,2),'k.-','linew',2)
hold on
plot(cont(:,1),cont(:,2),'r.-','linew',2)
xlabel('Time (sec.)'), ylabel('Stride time (s)')
legend({'Parkinson''s patient';'Control'})




% define sampling rate
srate  = 1000;

% create time series of steps
parkts = zeros(round(park(end,1)*1000),1);
parkts(round(park(2:end,1)*1000)) = 1;

% time vector and number of time points
parktx = (0:length(parkts)-1)/srate;
parkn  = length(parktx);

% repeat for control data
contts = zeros(round(cont(end,1)*1000),1);
contts(round(cont(2:end,1)*1000)) = 1;
conttx = (0:length(contts)-1)/srate;
contn  = length(conttx);


% plot the time course of steps
subplot(312)
stem(parktx,parkts,'ks')
xlabel('Time (sec.)'), ylabel('Step')
set(gca,'ylim',[-.1 1.1])


% compute power for both datasets
parkPow = 2*abs(fft(parkts)/parkn);
contPow = 2*abs(fft(contts)/contn);

% compute separate frequencies vector for each subject
parkHz = linspace(0,srate/2,floor(parkn/2)+1);
contHz = linspace(0,srate/2,floor(contn/2)+1);

% show power spectra
subplot(313)
plot(parkHz(2:end),parkPow(2:length(parkHz)),'k')
hold on
plot(contHz(2:end),contPow(2:length(contHz)),'r')
set(gca,'xlim',[0 7])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain')
legend({'Parkinson''s patient';'Control'})


% SOURCES:
%  Data downloaded from https://physionet.org/physiobank/database/gaitdb/
%   Parkinson's patient data is pd1-si.txt
%   Young control data is y1-23.si.txt

%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Applications of the Fourier transform
%      VIDEO: Rhythmicity in brain waves
% Instructor: mikexcohen.com
%
%%


% Load the data. If you get an error on this line,
% make sure MATLAB is in the same directory as the data.
% Or you can add the folder where the data are kept to the MATLAB path.
load EEGrestingState.mat

n = length(eegdata);
timevec = (0:n-1)/srate;

% compute amplitude spectrum
dataX    = fft(eegdata)/n;
ampspect = 2*abs(dataX);
hz       = linspace(0,srate/2,floor(n/2)+1);


figure(2), clf

% plot time domain signal
subplot(211)
plot(timevec,eegdata,'k')
xlabel('Time (sec.)'), ylabel('Amplitude (\muV)')
title('Time domain signal')



subplot(212)
% note: the following line uses the 'smooth' function, 
% which is in the curve-fitting toolbox. If you don't have 
% that function, comment that line and uncomment the next one.
plot(hz,smooth(ampspect(1:length(hz)),30),'k','linew',2)
%plot(hz,ampspect(1:length(hz)),'k','linew',2)
set(gca,'xlim',[0 70],'ylim',[0 .6])
xlabel('Frequency (Hz)'), ylabel('Amplitude (\muV)')
title('Frequency domain')


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Applications of the Fourier transform
%      VIDEO: Convolution theorem
% Instructor: mikexcohen.com
%
%%

m = 50; % length of signal
n = 11; % length of kernel

signal = zeros(m,1);
signal(round(m*.4):round(m*.6)) = 1;


kernel = zeros(n,1);
kernel(round(n*.25):round(n*.8)) = linspace(1,0,ceil(n*.55));


figure(3), clf

% plot signal
subplot(311)
plot(signal,'ks-','linew',2,'markerfacecolor','w')
title('Signal')

% plot kernel
subplot(312)
plot(kernel,'ks-','linew',2,'markerfacecolor','w')
set(gca,'xlim',[0 m])
title('Kernel')


% setup convolution parameters
nConv = m+n-1;
halfk = floor(n/2);

% convolution as point-wise multiplication of spectra and inverse
mx = fft(signal,nConv);
nx = fft(kernel,nConv);
% here's the convolution:
convres = ifft( mx.*nx );
% chop off the 'wings' of convolution
convres = convres(halfk+1:end-halfk);


% plot the result of convolution
subplot(313)
plot(convres,'rs-','linew',2,'markerfacecolor','w')
title('Result of convolution')


% for comparison, plot against the MATLAB convolution function
hold on
plot(conv(signal,kernel,'same'),'g','linew',2)



%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Applications of the Fourier transform
%      VIDEO: Narrowband temporal filtering
% Instructor: mikexcohen.com
%
%%

% Load the data. If you get an error on this line,
% make sure MATLAB is in the same directory as the data.
% Or you can add the folder where the data are kept to the MATLAB path.
load braindata.mat
n = length(timevec);

% plot time-domain signal
figure(4), clf
subplot(311)
plot(timevec,braindata,'k','linew',2)
set(gca,'xlim',[-.5 1.5])
xlabel('Time (sec.)'), ylabel('Voltage (\muV)')
title('Time domain')


% compute power spectrum
dataX    = fft(braindata);
ampspect = 2*abs(dataX/n).^2;
hz       = linspace(0,srate,n); % out to srate as trick for the filter

% plot power spectrum
subplot(312)
plot(hz,ampspect(1:length(hz)),'k','linew',2)
set(gca,'xlim',[0 100],'ylim',[0 500])
xlabel('Frequency (Hz)'), ylabel('Voltage (\muV)')
title('Frequency domain')



% specify which frequencies to filter
peakFiltFreqs = [2 47]; % Hz

c = 'kr'; % line colors
leglab = cell(size(peakFiltFreqs)); % legend entries


% loop over frequencies
for fi=1:length(peakFiltFreqs)
    
    % construct the filter
    x  = hz-peakFiltFreqs(fi); % shifted frequencies
    fx = exp(-(x/4).^2);       % gaussian
    
    % apply the filter to the data
    filtdat = 2*real( ifft( dataX.*fx ));
    
    % show the results
    subplot(313), hold on
    plot(timevec,filtdat,c(fi),'linew',2)
    set(gca,'xlim',[-.5 1.5])
    xlabel('Time (sec.)'), ylabel('Voltage (\muV)')
    title('Time domain')
    
    % line label for legend
    leglab{fi} = [ num2str(peakFiltFreqs(fi)) ' Hz' ];
end

% add legend
legend(leglab)


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Applications of the Fourier transform
%      VIDEO: Image smoothing and sharpening
% Instructor: mikexcohen.com
%
%%

%% with picture

% load image
lenna = imread('Lenna.png');
imgL  = double(mean(lenna,3));

figure(5), clf, colormap gray


% plot original image
subplot(221)
imagesc(imgL)
axis off, axis square
title('Original image')


% and its power spectrum
imgX  = fftshift(fft2(imgL));
powr2 = log(abs(imgX));

subplot(234)
imagesc(powr2)
set(gca,'clim',[0 15])
axis off, axis square
title('Amplitude spectrum')


% filter kernel is a Gaussian
width  = .1;   % width of gaussian (normalized Z units)
[x,y]  = ndgrid(zscore(1:size(imgL,1)),zscore(1:size(imgL,2)));

% add 1- at beginning of the next line to invert the filter
gaus2d = exp(-(x.^2 + y.^2) ./ (2*width^2)); 



subplot(235)
imagesc(gaus2d)
axis off, axis square
title('Gaussian (2D gain function)')


subplot(236)
imagesc( log(abs(imgX.*gaus2d)) )
axis off, axis square
set(gca,'clim',[0 15])
title('Modulated spectrum')


subplot(222)
imgrecon = real(ifft2( imgX.*gaus2d ));

imagesc( imgrecon )
axis off, axis square
title('Low-pass filtered image')


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Applications of the Fourier transform
%      VIDEO: Image narrowband filtering
% Instructor: mikexcohen.com
%
%%

%% example with two sine wave gradients

% specify vector of sine phases
sinephas = [ 0 pi/4 ];

% vector of sine frequencies
sinefreq = [.1 .05];  % arbitrary units


% sine wave initializations
lims  = [-91 91];
[x,y] = ndgrid(lims(1):lims(2),lims(1):lims(2));


% compute 2D sine gradients
xp = x*cos(sinephas(1)) + y*sin(sinephas(1));
img1 = sin( 2*pi*sinefreq(1)*xp );

xp = x*cos(sinephas(2)) + y*sin(sinephas(2));
img2 = sin( 2*pi*sinefreq(2)*xp );

% combine images
img = img1+img2;



figure(6), clf

% show original two gradients
subplot(321)
imagesc(img1), axis off, axis square
title('One image')

subplot(322)
imagesc(img2), axis off, axis square
title('Second image')


% show sum
subplot(323)
imagesc(img), axis off, axis square
title('Summed image')


% FFT
imgX    = fft2(img);
imgXamp = abs(imgX);

% show amplitude spectrum
subplot(324)
imagesc(fftshift(imgXamp))
set(gca,'clim',[0 500])
axis off, axis square
title('Amplitude spectrum')

% show sum down columns
subplot(325)
stem(sum(imgXamp),'ks'), axis square
title('Column sum of power spectra')

% replace 1st column with last
imgX(:,1) = imgX(:,end);

% reconstructed image
imgrecon  = real(ifft2(imgX));


subplot(326)
imagesc(imgrecon), axis square, axis off
title('Filtered image')

%% done.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
