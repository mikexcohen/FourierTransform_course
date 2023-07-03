%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete inverse Fourier transform
%      VIDEO: How and why it works
% Instructor: mikexcohen.com
%
%%


%% first, the forward Fourier transform

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
    fCoefs(fi) = sum( signal.*csw );
    % these are called the Fourier coefficients
end

% extract amplitudes
ampls = abs(fCoefs) / pnts;
ampls(2:end) = 2*ampls(2:end);

% compute frequencies vector
hz = linspace(0,srate/2,floor(pnts/2)+1);

figure(1), clf
plot(hz,ampls(1:length(hz)),'s-')
% better:
stem(hz,ampls(1:length(hz)),'ks-','linew',3,'markersize',10,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[0 10],'ylim',[-.01 3])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')

%% the inverse Fourier transform

% initialize time-domain reconstruction
reconSignal = zeros(size(signal));

for fi=1:pnts
    
    % create coefficient-modulated complex sine wave
    csw = fCoefs(fi) * exp( 1i*2*pi*(fi-1)*fourTime );
    
    % sum them together
    reconSignal = reconSignal + csw;
end

% divide by N
reconSignal = reconSignal/pnts;

figure(2), clf
plot(time,signal)
hold on
plot(time,real(reconSignal),'ro')
legend({'original';'reconstructed'})
xlabel('Time (s)')

%% one frequency at a time

clear

% set parameters
srate = 1000;
time  = 0:1/srate:3;
pnts  = length(time);

% create multispectral signal
signal  = (1+sin(2*pi*12*time)) .* cos(sin(2*pi*25*time)+time);

% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));

% here is the Fourier transform...
for fi=1:pnts
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    fCoefs(fi) = sum( signal.*csw ) / pnts;
end

% frequencies in Hz (goes up to srate just as a coding trick for later visualization)
hz = linspace(0,srate,pnts);


% setup plotting...
figure(3), clf
subplot(211)
plot(time,signal,'linew',2), hold on
sigh = plot(time,signal,'k','linew',2);
xlabel('Time (s)')
title('Time domain')

subplot(212)
powh = plot(1,'k','linew',2);
set(gca,'xlim',hz([1 end]),'xtick',0:100:900,'xticklabel',[0:100:500 400:-100:100])
title('Frequency domain')
xlabel('Frequencies (Hz)')


% initialize the reconstructed signal
reconSignal = zeros(size(signal));

% inverse Fourier transform here
for fi=1:pnts
    
    % create coefficient-modulated complex sine wave
    csw = fCoefs(fi) * exp( 1i*2*pi*(fi-1)*fourTime );
    
    % sum them together
    reconSignal = reconSignal + csw;
    
    % update plot for some frequencies
    if fi<dsearchn(hz',100) || fi>dsearchn(hz',srate-100)
        set(sigh,'YData',real(reconSignal)) % update signal
        set(powh,'XData',hz(1:fi),'YData',2*abs(fCoefs(1:fi))) % update amplitude spectrum
        pause(.05)
    end
end

%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: The discrete inverse Fourier transform
%      VIDEO: Inverse Fourier transform for bandstop filtering
% Instructor: mikexcohen.com
%
%%


%% bandpass filter

% simulation params
srate = 1000;
time  = 0:1/srate:2-1/srate;
pnts  = length(time);

% signal 
signal = sin(2*pi*4*time) + sin(2*pi*10*time);


% Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));
for fi=1:pnts
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    fCoefs(fi) = sum( signal.*csw ) / pnts;
end

% frequencies in Hz
hz = linspace(0,srate/2,floor(pnts/2)+1);

% find the coefficient for 10 Hz
freqidx = dsearchn(hz',10);

% set that coefficient to zero
fCoefsMod = fCoefs;
fCoefsMod(freqidx) = 0;


% and compute inverse
reconMod = zeros(size(signal));
for fi=1:pnts
    csw = fCoefsMod(fi) * exp( 1i*2*pi*(fi-1)*fourTime );
    reconMod = reconMod + csw;
end


% now plot
figure(4), clf

% plot original signal
subplot(311)
plot(time,signal,'k','linew',3)
title('Original signal, time domain')
xlabel('Time (s)')

% plot original amplitude spectrum
subplot(312)
stem(hz,2*abs(fCoefs(1:length(hz))),'ks-','linew',3,'markersize',14)
set(gca,'xlim',[0 25])
title('Original signal, frequency domain')
xlabel('Frequency (Hz)')


% and plot modulated time series
subplot(313)
plot(time,real(reconMod),'k','linew',3)
title('Band-stop filtered signal, time domain')
xlabel('Time (s)')

%% end.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
