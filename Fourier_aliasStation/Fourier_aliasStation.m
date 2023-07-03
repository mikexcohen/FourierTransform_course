%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Aliasing, stationarity, and violations
%      VIDEO: Aliasing
% Instructor: mikexcohen.com
%
%%


%% code for pictures shown in slides

srate = 1000;
t  = 0:1/srate:3;

sig = detrend( (1+sin(2*pi*2*t)) .* cos(sin(2*pi*5*t+cumsum(t/30))+t) );

figure(1), clf
subplot(311)
plot(t,sig,'w','linew',3)
axis off

subplot(312)
k = 25; % or 5
plot(t,sig,'w','linew',3), hold on
plot(t(1:k:end),sig(1:k:end),'rs','markerfacecolor','k','markersize',6)
axis off

subplot(313)
plot(t(1:k:end),sig(1:k:end),'r','linew',3)
axis off

set(gcf,'color',[33 33 33]/255)

%%

%  simulation parameters
srate  = 1000;
time   = 0:1/srate:1;
npnts  = length(time);
signal = sin(2*pi*5*time);


% measurement parameters
msrate = 6; % hz
mtime  = 0:1/msrate:1;
midx   = dsearchn(time',mtime');


% plot the time-domain signals
figure(2), clf
subplot(221)
plot(time,signal,'k','linew',3)
hold on
plot(time(midx),signal(midx),'mo','linew',2,'markerfacecolor','w','markersize',10)
set(gca,'ylim',[-1.1 1.1])
xlabel('time (s)'), ylabel('Amplitude')
title('Time domain')
legend({'"Analog"';'Sampled points'})


% plot the power spectrum of the "analog" signal
subplot(222)
sigX = 2*abs(fft(signal,npnts)/npnts);
hz   = linspace(0,srate/2,floor(npnts/2)+1);
stem(hz,sigX(1:length(hz)),'k','linew',2,'markerfacecolor','k')
set(gca,'xlim',[0 20])
xlabel('Frequency (Hz)'), ylabel('amplitude')
title('Frequency domain of "analog" signal')


% now plot only the measured signal
subplot(223)
plot(time(midx),signal(midx),'m','linew',2,'markerfacecolor','w','markersize',10)
set(gca,'ylim',[-1.1 1.1])
title('Measured signal')
xlabel('Time (s)'), ylabel('amplitude')

% and its amplitude spectrum
subplot(224)
sigX = 2*abs(fft(signal(midx),npnts)/length(midx));
hz   = linspace(0,msrate/2,floor(npnts/2)+1);
plot(hz,sigX(1:length(hz)),'k','linew',2,'markerfacecolor','k')
set(gca,'xlim',[0 20])
xlabel('Frequency (Hz)'), ylabel('amplitude')
title('Frequency domain of "analog" signal')

%% Related: getting close to the Nyquist

% subsample a high-sampling rate sine wave (pretend it's a continuous wave)
srate = 1000;
t = 0:1/srate:1;
f = 10; % frequency of the sine wave Hz
d = sin(2*pi*f*t);


% "Measurement" sampling rates
srates = [15 22 50 200]; % in Hz

figure(2), clf
for si=1:4
    subplot(2,2,si)
    
    % plot 'continuous' sine wave
    plot(t,d), hold on
    
    % plot sampled sine wave
    samples = round(1:1000/srates(si):length(t));
    plot(t(samples),d(samples),'ro-','linew',2)
    
    title([ 'Sampled at ' num2str(srates(si)/f) ' times' ])
    set(gca,'ylim',[-1.1 1.1],'xtick',0:.25:1)
end

%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Aliasing, stationarity, and violations
%      VIDEO: Effects of non-stationarities on power spectra
% Instructor: mikexcohen.com
%
%%


%% amplitude non-stationarity

srate = 1000;
t = 0:1/srate:10;
n = length(t);
f = 3; % frequency in Hz

% sine wave with time-increasing amplitude
ampl1 = linspace(1,10,n);
% uncomment this line for an AM-radio-like signal
% ampl1 = abs(interp1(linspace(t(1),t(end),10),10*rand(1,10),t,'spline'));
ampl2 = mean(ampl1);

signal1 = ampl1 .* sin(2*pi*f*t);
signal2 = ampl2 .* sin(2*pi*f*t);


% obtain Fourier coefficients and Hz vector
signal1X = fft(signal1)/n;
signal2X = fft(signal2)/n;
hz = linspace(0,srate/2,floor(n/2)+1);

figure(3), clf
subplot(211)
plot(t,signal2,'r','linew',2), hold on
plot(t,signal1,'b','linew',2)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('Time domain signal')

subplot(212)
stem(hz,2*abs(signal2X(1:length(hz))),'ro-','linew',2,'markerface','r')
hold on
stem(hz,2*abs(signal1X(1:length(hz))),'bs-','linew',2,'markerface','k')

title('Frequency domain')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[1 7])
legend({'Stationary';'Non-stationary'})

%% frequency non-stationarity

% create signals (sine wave and linear chirp)
f  = [2 10];
ff = linspace(f(1),mean(f),n);
signal1 = sin(2*pi.*ff.*t);
signal2 = sin(2*pi.*mean(ff).*t);

% Fourier spectra
signal1X = fft(signal1)/n;
signal2X = fft(signal2)/n;
hz = linspace(0,srate/2,floor(n/2));

figure(4), clf

% plot the signals in the time domain
subplot(211)
plot(t,signal1,'b'), hold on
plot(t,signal2,'r')
xlabel('Time (sec.)'), ylabel('Amplitude')
set(gca,'ylim',[-1.1 1.1])

% and their amplitude spectra
subplot(212)
stem(hz,2*abs(signal1X(1:length(hz))),'.-','linew',2), hold on
stem(hz,2*abs(signal2X(1:length(hz))),'r.-','linew',2)
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 20])

%% sharp transitions

a = [10 2 5 8];
f = [3 1 6 12];

timechunks = round(linspace(1,n,length(a)+1));

signal = 0;
for i=1:length(a)
    signal = cat(2,signal,a(i)* sin(2*pi*f(i)*t(timechunks(i):timechunks(i+1)-1) ));
end

signalX = fft(signal)/n;
hz = linspace(0,srate/2,floor(n/2)+1);

figure(5), clf
subplot(211)
plot(t,signal,'k')
xlabel('Time (s)'), ylabel('Amplitude')

subplot(212)
stem(hz,2*abs(signalX(1:length(hz))),'ks-','markerface','b')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 20])

%% phase reversal

srate = 1000;
ttime = 0:1/srate:1-1/srate; % temp time, for creating half the signal
time  = 0:1/srate:2-1/srate; % signal's time vector
n = length(time);

signal = [ sin(2*pi*10*ttime) -sin(2*pi*10*ttime) ];


figure(7), clf
subplot(211)
plot(time,signal,'k','linew',2)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('Time domain')


subplot(212)
signalAmp = (2*abs( fft(signal)/n )).^2;
hz = linspace(0,srate/2,floor(n/2)+1);
plot(hz,signalAmp(1:length(hz)),'ks-','linew',2,'markerfacecolor','w')
set(gca,'xlim',[5 15])
xlabel('Frequency (Hz)'), ylabel('Power')
title('Frequency domain')

%% edges and edge artifacts

x = (linspace(0,1,n)>.5)+0; % +0 converts from boolean to number

% uncommenting this line shows that nonstationarities 
% do not prevent stationary signals from being easily observed
% x = x + .08*sin(2*pi*6*time);

% plot
figure(6), clf
subplot(211)
plot(t,x,'k','linew',2)
set(gca,'ylim',[-.1 1.1])
xlabel('Time (s.)'), ylabel('Amplitude (a.u.)')

subplot(212)
xX = fft(x)/n;
stem(hz,2*abs(xX(1:length(hz))),'ks-')
set(gca,'xlim',[0 20],'ylim',[0 .1])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')

%% spike in the frequency domain

% frequency spectrum with a spike
fspect = zeros(300,1);
fspect(10) = 1;

% time-domain signal via iFFT
td_sig = real(ifft(fspect)) * length(fspect);


figure(8), clf

% plot amplitude spectrum
subplot(211)
stem(fspect,'ks-','linew',2)
xlabel('Frequency (indices)')
ylabel('Amplitude')
title('Frequency domain')

% plot time domain signal
subplot(212)
plot(td_sig,'k','linew',2)
xlabel('Time (indices)')
ylabel('Amplitude')
title('Time domain')


%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Aliasing, stationarity, and violations
%      VIDEO: Solutions for non-stationary time series
% Instructor: mikexcohen.com
%
%%

%% create signal (chirp) used in the following examples

% simulation details and create chirp
fs     = 1000; % sampling rate
time   = 0:1/fs:5;
npnts  = length(time);
f      = [10 30]; % frequencies in Hz
ff     = linspace(f(1),mean(f),npnts);
signal = sin(2*pi.*ff.*time);



figure(9), clf

% plot signal
subplot(211)
plot(time,signal,'k','linew',2)
xlabel('Time (s)'), ylabel('Amplitude')
title('Time domain signal')

% compute power spectrum
sigpow = 2*abs(fft(signal)/npnts).^2;
hz     = linspace(0,fs/2,floor(npnts/2)+1);

% and plot
subplot(212)
plot(hz,sigpow(1:length(hz)),'k','linew',2)
xlabel('Frequency (Hz)'), ylabel('Power')
set(gca,'xlim',[0 80])

%% short-time FFT

winlen   = 500; % window length
stepsize = 25;  % step size for STFFT
numsteps = floor( (npnts-winlen)/stepsize );

hz = linspace(0,fs/2,floor(winlen/2)+1);


% initialize time-frequency matrix
tf = zeros(length(hz),numsteps);

% Hann taper
hwin = .5*(1-cos(2*pi*(1:winlen) / (winlen-1)));

% loop over time windows
for ti=1:numsteps
    
    % extract part of the signal
    tidx    = (ti-1)*stepsize+1:(ti-1)*stepsize+winlen;
    tapdata = signal(tidx);
    
    % FFT of these data
    x = fft(hwin.*tapdata)/winlen;
    
    % and put in matrix
    tf(:,ti) = 2*abs(x(1:length(hz)));
end

figure(10), clf
subplot(211)
contourf(time( (0:numsteps-1)*stepsize+1 ),hz,tf,40,'linecolor','none')
set(gca,'ylim',[0 50],'xlim',[0 5],'clim',[0 .5])
xlabel('Time (s)'), ylabel('Frequency (Hz)')
title('Time-frequency power via short-time FFT')
colorbar

%% Morlet wavelet convolution

% frequencies used in analysis
nfrex = 30;
frex  = linspace(2,50,nfrex);
wtime = -2:1/fs:2;
gausS = linspace(5,35,nfrex);

% convolution parameters
nConv = length(wtime) + npnts - 1;
halfw = floor(length(wtime)/2);

% initialize time-frequency output matrix
tf = zeros(nfrex,npnts);

% FFT of signal
signalX = fft(signal,nConv);

% loop over wavelet frequencies
for fi=1:nfrex
    
    % create the wavelet
    s   = ( gausS(fi)/(2*pi*frex(fi)) )^2;
    cmw = exp(1i*2*pi*frex(fi)*wtime) .* exp( (-wtime.^2)/s );
    
    % compute its Fourier spectrum and normalize
    cmwX = fft(cmw,nConv);
    cmwX = cmwX./max(cmwX); % scale to 1
  
    
    % convolution result is inverse FFT of pointwise multiplication
    convres  = ifft( signalX .* cmwX );
    tf(fi,:) = 2*abs(convres(halfw+1:end-halfw));
end

subplot(212)
contourf(time,frex,tf,40,'linecolor','none')
set(gca,'ylim',[0 50],'xlim',[0 5],'clim',[0 1])
xlabel('Time (s)'), ylabel('Frequency (Hz)')
title('Time-frequency power via complex Morlet wavelet convolution')
colorbar


%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Aliasing, stationarity, and violations
%      VIDEO: Windowing and Welch's method
% Instructor: mikexcohen.com
%
%%

% create signal
srate = 1000;
npnts = 2000; % actually, this times 2!
time  = (0:npnts*2-1)/srate;
freq  = 10; % Hz

% create signal
signal = [sin(2*pi*freq*time(1:npnts)) sin(2*pi*freq*time(1:npnts) + pi)];

% compute its amplitude spectrum
hz = linspace(0,srate/2,floor(length(time)/2)+1);
ampspect = abs(fft(signal)/length(time));
ampspect(2:length(hz)-1) = 2*ampspect(2:length(hz)-1);


figure(4), clf
% plot the time-domain signal
subplot(311)
plot(time,signal)
xlabel('Time (s)'), ylabel('Amplitude')
title('Time domain')


% plot the frequency domain signal
subplot(312)
stem(hz,ampspect(1:length(hz)),'ks-','linew',2)
set(gca,'xlim',[0 freq*2])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain (FFT)')

%% now for Welch's method

% parameters
winlen = 1000; % window length in points (not ms!)
nbins = floor(length(time)/winlen);

% vector of frequencies for the small windows
hzL = linspace(0,srate/2,floor(winlen/2)+1);

% initialize time-frequency matrix
welchspect = zeros(1,length(hzL));

% Hann taper
hwin = .5*(1-cos(2*pi*(1:winlen) / (winlen-1)));

% loop over time windows
for ti=1:nbins
    
    % extract part of the signal
    tidx    = (ti-1)*winlen+1:ti*winlen;
    tmpdata = signal(tidx);
    
    % FFT of these data (does the taper help?)
    x = fft(hwin.*tmpdata)/winlen;
    
    % and put in matrix
    welchspect = welchspect + 2*abs(x(1:length(hzL)));
end

% divide by nbins to complete average
welchspect = welchspect/nbins;

subplot(313)
stem(hzL,welchspect,'ks-','linew',2)
set(gca,'xlim',[0 freq*2])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain (Welch''s method)')

%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Aliasing, stationarity, and violations
%      VIDEO: Instantaneous frequency
% Instructor: mikexcohen.com
%
%%

% simulation details
srate = 1000;
time  = 0:1/srate:4;
pnts  = length(time);

% frequencies for Fourier transform
hz = linspace(0,srate/2,floor(length(time)/2)-1);


% frequency range for linear chirp
f  = [7 25];

% generate chirp signal
ff = linspace(f(1),mean(f),pnts);
signal = sin(2*pi.*ff.*time);

% instantaneous frequency
angels = angle(hilbert(signal));
instfreq = diff(unwrap(angels))/(2*pi/srate);

subplot(311)
plot(time,signal,'k','linew',2)
xlabel('Time (s)'), ylabel('Amplitude')
title('Time domain')

subplot(312)
amp = 2*abs(fft(signal)/pnts);
plot(hz,amp(1:length(hz)),'ks-','linew',2,'markerfacecolor','w')
title('Frequency domain')
set(gca,'xlim',[0 min(srate/2,f(2)*3)])
xlabel('Frequency (Hz)'), ylabel('Amplitude')

subplot(313)
plot(time(1:end-1),instfreq,'k','linew',2,'markerfacecolor','w')
set(gca,'ylim',[f(1)*.8 f(2)*1.2])
xlabel('Time (s)'), ylabel('Frequency (Hz)')
title('Instantaneous frequency')


%% done.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
