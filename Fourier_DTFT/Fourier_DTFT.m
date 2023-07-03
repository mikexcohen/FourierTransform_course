%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: How it works
% Instructor: mikexcohen.com
%
%%

%% The DTFT in loop-form

% create the signal
srate  = 1000; % hz
time   = 0:1/srate:2; % time vector in seconds
pnts   = length(time); % number of time points
signal = 2.5 * sin( 2*pi*4*time ) ...
       + 1.5 * sin( 2*pi*6.5*time );


% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));

for fi=1:pnts
    
    % create complex sine wave
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    
    % compute dot product between sine wave and signal
    fCoefs(fi) = sum( signal.*csw ) / pnts;
    % these are called the Fourier coefficients
end

% extract amplitudes
ampls = 2*abs(fCoefs);

% compute frequencies vector
hz = linspace(0,srate/2,floor(pnts/2)+1);

figure(1), clf
subplot(211)
plot(time,signal,'k','linew',2)
xlabel('Time (s)'), ylabel('Amplitude')
title('Time domain')

subplot(212)
plot(hz,ampls(1:length(hz)),'ks-','linew',3,'markersize',15,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[0 10],'ylim',[-.01 3])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')
title('Frequency domain')



% now plot MATLAB's fft output on top
fCoefsF = fft(signal)/pnts;
amplsF  = 2*abs(fCoefsF);
hold on
% plot(hz,amplsF(1:length(hz)),'ro','markerfacecolor','r')

%% plot two Fourier coefficients

coefs2plot = dsearchn(hz',[4 4.5]');

% extract magnitude and angle
mag = abs(fCoefs(coefs2plot));
phs = angle(fCoefs(coefs2plot));

figure(2), clf
plot( real(fCoefs(coefs2plot)) , imag(fCoefs(coefs2plot)) ,'o','linew',2,'markersize',10,'markerfacecolor','r');

% make plot look nicer
axislims = max(mag)*1.1;
set(gca,'xlim',[-1 1]*axislims,'ylim',[-1 1]*axislims)
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Real axis')
ylabel('Imaginary axis')
title('Complex plane')


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Converting indices into frequencies
% Instructor: mikexcohen.com
%
%%

pnts     = 16; % number of time points
fourTime = (0:pnts-1)/pnts;
figure(2), clf

for fi=1:pnts
    % create complex sine wave
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    
    % and plot it
    subplot(4,4,fi)
    h = plot(fourTime,real(csw), fourTime,imag(csw),'linew',2);
    
    % adjust line colors
    set(h(1),'color','k')
    set(h(2),'color',[1 1 1]*.5)
    
    % adjust the plot settings
    set(gca,'ylim',[-1 1]*1.1,'ytick',[],'xtick',[],'xlim',fourTime([1 end]))
    axis square
    title([ 'freq. idx ' num2str(fi-1) ])
end
legend({'Real';'Imag'})


    
%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Converting indices to frequencies, part 2
% Instructor: mikexcohen.com
%
%%

%% 

% some parameters
srate = 1000;
npnts = 1200;

% frequencies vector
if mod(npnts,2)==0
  topfreq = srate/2;
else
  topfreq = srate/2 * (npnts-1)/npnts;
end
hz1 = linspace(0,srate/2,floor(npnts/2+1));
hz2 = linspace(0,topfreq,floor(npnts/2+1));

% some arbitary frequency to show
n = 36;
fprintf('\n\n\n%.9f\n%.9f\n',hz1(n),hz2(n))


    
%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Shortcut: converting indices to frequencies
% Instructor: mikexcohen.com
%
%%

%% Case 1: ODD number of data points, N is correct

% create the signal
srate = 1000;
time  = (0:srate)/srate;
npnts = length(time);

% Notice: A simple 15-Hz sine wave!
signal = sin(15*2*pi*time);

% its amplitude spectrum
signalX = 2*abs(fft(signal)) / length(signal);

% frequencies vector
hz1 = linspace(0,srate,npnts+1);
hz2 = linspace(0,srate,npnts);


% plot it
figure(1), clf, hold on
stem(hz1(1:npnts),signalX,'bs-','linew',3,'markersize',12)
stem(hz2,signalX,'ro-','linew',3,'markersize',12)
axis([14.9 15.1 .99 1.01])
title([ num2str(length(time)) ' points' ])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
legend({'N+1';'N'})


%% Case 2: EVEN number of data points, N+1 is correct

% create the signal
srate = 1000;
time  = (0:srate-1)/srate;
npnts = length(time);

% Notice: A simple 15-Hz sine wave!
signal = sin(15*2*pi*time);

% its amplitude spectrum
signalX = 2*abs(fft(signal)) / length(signal);

% frequencies vector
hz1 = linspace(0,srate,npnts+1);
hz2 = linspace(0,srate,npnts);


% plot it
figure(2), clf, hold on
stem(hz1(1:npnts),signalX,'bs-','linew',3,'markersize',12)
stem(hz2,signalX,'ro-','linew',3,'markersize',12)
axis([14.9 15.1 .99 1.01])
title([ num2str(length(time)) ' points' ])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
legend({'N+1';'N'})



%% Case 3: longer signal

% create the signal
srate = 1000;
time  = (0:5*srate-1)/srate;
npnts = length(time);

% Notice: A simple 15-Hz sine wave!
signal = sin(15*2*pi*time);

% its amplitude spectrum
signalX = 2*abs(fft(signal)) / length(signal);

% frequencies vector
hz1 = linspace(0,srate,npnts+1);
hz2 = linspace(0,srate,npnts);


% plot it
figure(3), clf, hold on
stem(hz1(1:npnts),signalX,'bs-','linew',3,'markersize',12)
stem(hz2,signalX,'ro-','linew',3,'markersize',12)
axis([14.99 15.01 .99 1.01])
title([ num2str(length(time)) ' points' ])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
legend({'N+1';'N'})


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Normalized time vector
% Instructor: mikexcohen.com
%
%%

% create the signal
srate  = 1000; % hz
time   = 0:1/srate:2; % time vector in seconds
pnts   = length(time); % number of time points
signal = 2.5 * sin( 2*pi*4*time ) ...
       + 1.5 * sin( 2*pi*6.5*time );


% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));

for fi=1:pnts
    
    % create complex sine wave
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    
    % compute dot product between sine wave and signal
    fCoefs(fi) = sum( signal.*csw ) / pnts;
    % these are called the Fourier coefficients
end

% extract amplitudes
ampls = 2*abs(fCoefs);

% compute frequencies vector
hz = linspace(0,srate/2,floor(pnts/2)+1);

figure(1), clf
subplot(311)
plot(time,signal,'k','linew',2)
xlabel('Time (s)'), ylabel('Amplitude')
title('Time domain')

% plot amplitude
subplot(312)
stem(hz,ampls(1:length(hz)),'ks-','linew',3,'markersize',15,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[0 10],'ylim',[-.01 3])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')
title('Amplitude spectrum')

% plot angles
subplot(313)
stem(hz,angle(fCoefs(1:length(hz))),'ks-','linew',3,'markersize',15,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[0 10],'ylim',[-1 1]*pi)
xlabel('Frequency (Hz)'), ylabel('Phase (rad.)')
title('Phase spectrum')


% finally, plot reconstructed time series on top of original time series
reconTS = real(ifft( fCoefs ))*pnts;
subplot(311), hold on
plot(time(1:3:end),reconTS(1:3:end),'r.')
legend({'Original';'Reconstructed'})

% inspect the two time series. they should be identical!
zoom on 


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Accurate scaling of Fourier coefficients
% Instructor: mikexcohen.com
%
%%


%% incorrect amplitude units without normalizations

% create the signal
srate  = 1000; % hz
time   = 0:1/srate:4.5; % time vector in seconds
pnts   = length(time); % number of time points
signal = 4 * sin( 2*pi*4*time );


% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));

for fi=1:pnts
    % create complex sine wave and compute dot product with signal
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    fCoefs(fi) = sum( signal.*csw );
end

fCoefs = fCoefs/pnts;

% extract amplitudes
ampls = 2*abs(fCoefs);

% compute frequencies vector
hz = linspace(0,srate/2,floor(pnts/2)+1);

figure(3), clf
stem(hz,ampls(1:length(hz)),'ks-','linew',3,'markersize',10,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[0 10])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Interpreting phase values
% Instructor: mikexcohen.com
%
%%


%% same amplitude, different phase

% simulation parameters
srate = 1000;
time  = 0:1/srate:1;
npnts = length(time);

% generate signal
signal1 = 2.5*sin(2*pi*10*time +   0  ); % different phase values
signal2 = 2.5*sin(2*pi*10*time + pi/2 ); 

% Fourier transform
fourTime = (0:npnts-1)/npnts;
signal1X = zeros(size(signal1));
signal2X = zeros(size(signal2));

for fi=1:npnts
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    signal1X(fi) = sum( signal1.*csw );
    signal2X(fi) = sum( signal2.*csw );
end

% frequencies vector
hz = linspace(0,srate/2,floor(length(time)/2)+1);


% extract correctly-normalized amplitude
signal1Amp = abs(signal1X(1:length(hz))/npnts);
signal1Amp(2:end) = 2*signal1Amp(2:end);

signal2Amp = abs(signal2X(1:length(hz))/npnts);
signal2Amp(2:end) = 2*signal2Amp(2:end);


% now extract phases
signal1phase = angle(signal1X(1:length(hz)));
signal2phase = angle(signal2X(1:length(hz)));


figure(5), clf

% plot time-domain signals
subplot(321), plot(time,signal1,'k')
title('Signal 1, time domain')
xlabel('Time (s)'), ylabel('Amplitude')

subplot(322), plot(time,signal2,'k')
title('Signal 2, time domain')
xlabel('Time (s)'), ylabel('Amplitude')

subplot(323), stem(hz,signal1Amp,'k')
title('Frequency domain')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 20],'ylim',[0 2.8])

subplot(324), stem(hz,signal2Amp,'k')
title('Frequency domain')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 20],'ylim',[0 2.8])

subplot(325), stem(hz,signal1phase,'k')
xlabel('Frequency (Hz)'), ylabel('Phase (rad.)')
set(gca,'xlim',[0 20])

subplot(326), stem(hz,signal2phase,'k')
xlabel('Frequency (Hz)'), ylabel('Phase (rad.)')
set(gca,'xlim',[0 20])


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Averaging Fourier coefficients
% Instructor: mikexcohen.com
%
%%

% simulation parameters
ntrials = 100;
srate   = 200; % Hz
time    = 0:1/srate:1-1/srate;
pnts    = length(time);


% create dataset
data = zeros(ntrials,pnts);
for triali=1:ntrials
    data(triali,:) = sin(2*pi*20*time + 2*pi*rand);
end

% plot the data
figure(6), clf
subplot(211), hold on
plot(time,data);
plot(time,mean(data),'k','linew',3)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('Time domain')
% set(gca,'xlim',[0 .1])

% get Fourier coefficients
dataX = fft(data,[],2) / pnts;
hz = linspace(0,srate/2,floor(pnts/2)+1);

% averaging option 1: complex Fourier coefficients, then magnitude
ave1 = 2*abs( mean(dataX) );

% averaging option 2: magnitude, then complex Fourier coefficients
ave2 = mean( 2*abs(dataX) );

% plot both amplitude spectra
subplot(212), hold on
stem(hz,ave1(1:length(hz)),'ks','linew',3,'markersize',10,'markerfacecolor','w')
stem(hz+.2,ave2(1:length(hz)),'ro','linew',3,'markersize',10,'markerfacecolor','w')
set(gca,'xlim',[10 30])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain')
legend({'Average coefficients';'Average amplitude'})

%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: The DC (zero frequency) component
% Instructor: mikexcohen.com
%
%%


%% incorrect DC reconstruction without careful normalization

% create the signal
srate  = 1000; % hz
time   = 0:1/srate:2; % time vector in seconds
pnts   = length(time); % number of time points
signal =  1.5 + 2.5*sin( 2*pi*4*time );


% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));

for fi=1:pnts
    % create complex sine wave and compute dot product with signal
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    fCoefs(fi) = sum( signal.*csw );
end

% extract amplitudes
ampls = 2*abs(fCoefs/pnts);

% compute frequencies vector
hz = linspace(0,srate/2,floor(pnts/2)+1);

figure(7), clf
stem(hz,ampls(1:length(hz)),'ks-','linew',3,'markersize',10,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[-.1 10])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')


%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete Fourier transform
%      VIDEO: Amplitude spectrum vs. power spectrum
% Instructor: mikexcohen.com
%
%%

% simulation parameters
srate = 1000;
time  = 0:1/srate:.85;
npnts = length(time);

% generate signal
signal = 2.5*sin(2*pi*10*time);

% Fourier transform
fourTime = (0:npnts-1)/npnts;
signalX  = zeros(size(signal));
for fi=1:npnts
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    signalX(fi) = sum( signal.*csw );
end

% frequencies vector
hz = linspace(0,srate/2,floor(length(time)/2)+1);


% extract correctly-normalized amplitude
signalAmp = abs(signalX(1:length(hz))/npnts);
signalAmp(2:end) = 2*signalAmp(2:end);

% and power
signalPow = signalAmp.^2;


figure(8), clf

% plot time-domain signal
subplot(311)
plot(time,signal,'k','linew',2)
xlabel('Time (ms)')
ylabel('Amplitude')
title('Time domain')

% plot frequency domain spectra
subplot(312)
plot(hz,signalAmp,'ks-','linew',2,'markerfacecolor','w','markersize',10)
hold on
plot(hz,signalPow,'rs-','linew',2,'markerfacecolor','w','markersize',10)

set(gca,'xlim',[0 20])
legend({'Amplitude';'Power'})
xlabel('Frequency (Hz)')
ylabel('Amplitude or power')
title('Frequency domain')

% plot dB power
subplot(313)
plot(hz,10*log10(signalPow),'ks-','linew',2,'markerfacecolor','w','markersize',10)
set(gca,'xlim',[0 20])
xlabel('Frequency (Hz)')
ylabel('Decibel power')

%% done.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
