%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: The fast Fourier transform
%      VIDEO: How the FFT works, speed tests
% Instructor: mikexcohen.com
%
%%

% create the signal
pnts   = 10000; % number of time points
signal = randn(1,pnts);


% start the timer for "slow" Fourier transform
tic;

% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));

for fi=1:pnts
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    fCoefs(fi) = sum( signal.*csw );
end

% end timer for slow Fourier transform
t(1) = toc;


% time the fast Fourier transform
tic;
fCoefsF = fft(signal);
t(2) = toc;

% and plot
figure(1), clf
bar(t)
set(gca,'xlim',[0 3],'xticklabel',{'slow';'fast'},'xtick',1:2)
ylabel('Computation time (sec.)')

%% fft still need normalizations

srate = 1000;
time  = 0:1/srate:2;
npnts = length(time);

% signal
signal = 2*sin(2*pi*6*time);

% Fourier spectrum
signalX = fft(signal);
hz = linspace(0,srate/2,floor(npnts/2)+1);

% amplitude
ampl = abs(signalX(1:length(hz)));

figure(2), clf
stem(hz,ampl,'ks-','linew',3,'markersize',10,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[0 10])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')

%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: The fast Fourier transform
%      VIDEO: The fast inverse Fourier transform
% Instructor: mikexcohen.com
%
%%

% set parameters
srate = 1000;
time  = 0:1/srate:3;

% create multispectral signal
signal  = (1+sin(2*pi*12*time)) .* cos(sin(2*pi*25*time)+time);

% fft
signalX = fft(signal);

% reconstruction via ifft
reconSig = ifft(signalX);

% could also be done in one line:
%reconSig = ifft(fft(signal));

figure(3), clf
plot(time,signal,'b.-','linew',2,'markersize',15)
hold on
plot(time,reconSig,'ro','linewidth',2)
xlabel('Time (sec.)'), ylabel('amplitude (a.u.)')
legend({'Original';'Reconstructed'})
zoom on


%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: The fast Fourier transform
%      VIDEO: The perfection of the Fourier transform
% Instructor: mikexcohen.com
%
%%

pnts = 1000; % number of time points

% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
F        = zeros(pnts);

for fi=1:pnts
    
    % create complex sine wave
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    
    % put csw into column of matrix F
    F(:,fi) = csw;
end

% compute inverse of F (and normalize by N)
Finv = inv(F)*pnts;

% plot one sine wave
figure(4), clf
subplot(211)
plot(fourTime,real(F(:,5)), fourTime,imag(F(:,5)),'linew',3)
xlabel('Time (norm.)')
legend({'real';'imag'})
set(gca,'ylim',[-1.05 1.05])
title('One column of matrix F')

subplot(212)
plot(fourTime,real(Finv(:,5)), fourTime,imag(Finv(:,5)),'linew',3)
xlabel('Time (norm.)')
legend({'real';'imag'})
set(gca,'ylim',[-1.05 1.05])
title('One column of matrix F^{-1}')

%%
%     COURSE: Signal processing and image processing in MATLAB and Python
%    SECTION: The fast Fourier transform
%      VIDEO: Using the fft on matrices
% Instructor: mikexcohen.com
%
%%


% generate multivariate dataset
srate = 400;
time  = (0:srate*2-1)/srate;
npnts = length(time);
nreps = 50;

% dataset is repeated sine waves
data = repmat( sin(2*pi*10*time), nreps,1 );

% FFT of data along each dimension
dataX1 = fft(data,[],1) / npnts;
dataX2 = fft(data,[],2) / npnts;
hz = linspace(0,srate/2,floor(npnts/2)+1);

% check sizes
size(dataX1)
size(dataX2)

% show data and spectra!
figure(5), clf
subplot(311)
imagesc(data)
xlabel('Time'), ylabel('Channel')
title('Time-domain signal')

subplot(312)
stem(hz,mean( 2*abs(dataX1(:,1:length(hz))) ,1),'k')
xlabel('Frequency (??)'), ylabel('Amplitude')
title('FFT over channels')

subplot(313)
stem(hz,mean( 2*abs(dataX2(:,1:length(hz))) ,1),'k')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('FFT over time')

%% done.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
