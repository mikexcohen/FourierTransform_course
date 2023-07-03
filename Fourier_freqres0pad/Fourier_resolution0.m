%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Frequency resolution and zero padding
%      VIDEO: Sampling and frequency resolution
% Instructor: mikexcohen.com
%
%%

% create the signal
srate  = 1000;
pnts   = 1000;

% The Fourier transform loop (although it's not being computed)
for fi=1:pnts
    %csw = exp( -1i*2*pi*(fi-1)*fourTime );
    %fCoefs(fi) = sum( signal.*csw ) / pnts;
end

% compute frequencies vector
hz = linspace(0,srate/2,floor(pnts/2)+1);

% compute empirical frequency resolution as spacing between frequencies
freqres = mean(diff(hz));

% ... or directly from sampling rate and N
freqres = srate/pnts;


% print result in command window
fprintf('Frequency resolution is %g Hz\n',freqres)

%% example of increasing the number of data points

% parameters (try adjusting the srate and end of time)
srate  = 10;
time   = 0:1/srate:2;

% create signal
signal = zeros(size(time));
signal(1:round(length(time)*.1)) = 1;

% spectrum and frequencies vector
signalX = fft(signal);
hz = linspace(0,srate,length(time)); % plotting trick... frequencies really only go up to Nyquist

% plot
figure(1), clf
subplot(211)
plot(time,signal,'ks-','linew',2,'markerfacecolor','w','markersize',10)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('Time domain')


subplot(212)
plot(hz,2*abs(signalX),'ks-','linew',2,'markerfacecolor','w','markersize',10)
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title([ 'Frequency domain (resolution = ' num2str(diff(hz(1:2))) ' Hz)' ])

%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Frequency resolution and zero padding
%      VIDEO: Time-domain zero padding
% Instructor: mikexcohen.com
%
%%

%% signal

% create the signal (a Hann window)
n = 40;
signal = sin( (0:n-1) * pi/(n-1) ).^2;

% take the fast Fourier transform
signalX = fft(signal);

% extract amplitude
ampl = abs(signalX);

% normalized frequency units
frequnits = linspace(0,1,length(signal));

% and plot
figure(2), clf
subplot(221)
plot(signal,'ks-','linew',2,'markerfacecolor','w','markersize',10)
set(gca,'xlim',[0 80])
xlabel('Time (a.u.)'), ylabel('Amplitude')
title('Time domain')

subplot(223)
stem(frequnits,ampl,'ks-','linew',2,'markersize',10,'markerfacecolor','w')
set(gca,'xlim',[-.01 .3])
xlabel('Frequency (a.u.)'), ylabel('Amplitude')
title('Frequency domain')

%% manually add zeros

signalPad = cat(2,signal,zeros(1,40));

% Fourier coefficients
signalX = fft(signalPad);

% extract amplitude
ampl = abs(signalX);

% normalized frequency units
frequnits = linspace(0,1,length(signalPad));

% and plot
subplot(222)
plot(signalPad,'ks-','linew',2,'markerfacecolor','w','markersize',10)
xlabel('Time (a.u.)'), ylabel('Amplitude')
set(gca,'xlim',[0 80])
title('Time domain, zero-padded')

subplot(224)
stem(frequnits,ampl,'ks-','linew',2,'markersize',10,'markerfacecolor','w')
set(gca,'xlim',[-.01 .3])
xlabel('Frequency (a.u.)'), ylabel('Amplitude')
title('Frequency domain, zero-padded')

%% zero-padding in fft function

% create the signal
signal = [ 4 6 -1 0 5 -4 ];

% number of zeros to add after signal
n2pad = [ 0 10 100 ];

figure(3), clf

c = 'krb';
s = 'o^s';

leg = cell(size(n2pad));
for zi=1:length(n2pad)
    
    % total length of signal
    zeropadN = length(signal)+n2pad(zi);
    
    % FFT and amplitude
    sigampl   = abs( fft(signal,zeropadN) );
    
    % one of the two normalization steps
    sigampl = sigampl / 1;
    
    frequnits = linspace(0,1,zeropadN+1);
    
    % and plot
    plot(frequnits(1:length(sigampl)),sigampl,[c(zi) s(zi) '-'],'linew',2,'markerfacecolor','w','markersize',10)
    hold on
    leg{zi} = [ num2str(zeropadN) '-point FFT' ];
end

% add some plot extras
legend(leg)
xlabel('Frequency (.5 = Nyquist)')
ylabel('Amplitude (a.u.)')


%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Frequency resolution and zero padding
%      VIDEO: Frequency-domain zero padding
% Instructor: mikexcohen.com
%
%%


% create the signal
signal = [ 4 6 -1 0 5 -4 ];

% its Fourier coefficients
signalX = fft(signal);


% number of zeros to add after spectrum
n2pad = [ 0 10 100 ];

figure(4), clf, hold on

c = 'krb';
s = 'o^s';

for zi=1:length(n2pad)
    
    % spectral zero-padding
    zeropadN = length(signal)+n2pad(zi);
    
    % reconstruction via ifft
    reconSig = ifft(signalX,zeropadN) * zeropadN;
    normtime = linspace(0,1,length(reconSig));
    
    % and plot
    plot(normtime,real(reconSig),[c(zi) s(zi) '-'],'linew',2,'markerfacecolor','w','markersize',10)
    
    % legend
    leg{zi} = [ num2str(zeropadN) '-point IFFT' ];
end

% add some plot extras
legend(leg)
xlabel('Time (norm.)')

%% Another example

% create the signal
srate  = 1000;
x      = (0:255)/srate;
signal = sin(2*pi*20*x)>.3;

% number of zeros to add after spectrum
n2pad = [ 1 2 5 ];

figure(5), clf

c = 'krb';
s = 'o^s';

leg = cell(size(n2pad));
for zi=1:length(n2pad)
    
    % fft
    signalX = fft(signal);
    
    % spectral zero-padding
    zeropadN = 2^nextpow2( length(signal)*n2pad(zi) );
    
    % reconstruction via ifft
    reconSig = ifft(signalX,zeropadN) * zeropadN;
    normtime = linspace(x(1),x(end),length(reconSig)); % new time vector
    srateNew = 1/mean(diff(normtime)); % new sampling rate
    
    % power spectrum
    ampl = abs(fft( real(reconSig )));
    hz   = linspace(0,srateNew/2,floor(length(reconSig)/2+1));
    
    
    % plot time-domain signal
    subplot(211)
    if zi==1
        plot(normtime,real(reconSig),[c(zi) 's-'],'linew',2,'markersize',10,'markerfacecolor','w')
    else
        plot(normtime,real(reconSig),[c(zi) '.-'],'markersize',15)
    end
    hold on, axis tight
    leg{zi} = [ num2str(zeropadN) '-point IFFT' ];
    
    % plot amplitude spectrum
    subplot(212)
    plot(hz,ampl(1:length(hz)),[c(zi) '.-']), hold on
end

% add some plot extras
legend(leg)
subplot(211)
xlabel('Time (sec.)')
title('Time domain')

subplot(212)
xlabel('Frequency (Hz)'), ylabel('Amplitude (non-normalized)')
title('Frequency domain')
zoom on, axis tight


%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: Frequency resolution and zero padding
%      VIDEO: Sampling rate vs. signal length
% Instructor: mikexcohen.com
%
%%


%% difference between sampling rate and number of time points for Fourier frequencies

% temporal parameters
srates  = [100 100 1000];
timedur = [  1  10    1];


freq    =     5; % in Hz
colors  = 'kmb';
symbols = 'op.';


figure(6), clf
legendText = cell(size(timedur));
for parami=1:length(colors)
    
    % define sampling rate in this round
    srate = srates(parami); % in Hz
    
    % define time
    time = -1:1/srate:timedur(parami);
    
    % create signal (Morlet wavelet)
    signal = cos(2*pi*freq.*time) .* exp( (-time.^2) / .05 );
    
    % compute FFT and normalize
    signalX = fft(signal);
    signalX = signalX./max(signalX);
    
    % define vector of frequencies in Hz
    hz = linspace(0,srate/2,floor(length(signal)/2)+1);
    
    
    % plot time-domain signal
    subplot(211)
    plot(time,signal,[colors(parami) symbols(parami) '-'],'markersize',10,'markerface',colors(parami)), hold on
    set(gca,'xlim',[-1 3])
    xlabel('Time (s)'), ylabel('Amplitude')
    title('Time domain')
    
    % plot frequency-domain signal
    subplot(212), hold on
    plot(hz,abs(signalX(1:length(hz))),[colors(parami) symbols(parami) '-'],'markersize',10,'markerface',colors(parami))
    xlabel('Frequency (Hz)'), ylabel('Amplitude')
    title('Frequency domain')
    
    legendText{parami} = [ 'srate=' num2str(srates(parami)) ', N=' num2str(timedur(parami)+1) 's' ];
end

legend(legendText)
zoom on

%% done.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
