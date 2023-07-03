%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Introductions
%      VIDEO: Pitch
% Instructor: mikexcohen.com
%
%%

srate = 1000;
time = 0:1/srate:3;

x = (1+sin(2*pi*2*time)) .* cos(sin(2*pi*5*time)+time);
xx = 2*abs(fft(x)/length(x));

figure(1), clf
subplot(211)
plot(time,x,'k','linew',3)
xlabel('Time (sec.)')
ylabel('Amplitude')
title('Time domain')


subplot(212)
hz = linspace(0,srate,length(x));
stem(hz,xx.^2,'k','linew',3,'markersize',5,'markerfacecolor','k')
set(gca,'xlim',[0 30])
xlabel('Frequencies (Hz)')
ylabel('Power')
title('Frequency domain')

%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Introductions
%      VIDEO: Examples of Fourier transform applications
% Instructor: mikexcohen.com
%
%%

%% 1D examples

srate = 1000;
time  = 0:1/srate:2-1/srate;
n     = length(time);
hz    = linspace(0,srate,n);

%%% pure sine wave
signal = sin(2*pi*5*time);

% %%% multispectral wave
% signal = 2*sin(2*pi*5*time) + 3*sin(2*pi*7*time) + 6*sin(2*pi*14*time);
% 
% %%% white noise
% signal = randn(size(time));
% 
% %%% Brownian noise (aka random walk)
% signal = cumsum(randn(size(time)));
% 
% %%% 1/f noise
% ps   = exp(1i*2*pi*rand(1,n/2)) .* .1+exp(-(1:n/2)/50);
% ps   = [ps ps(:,end:-1:1)];
% signal = real(ifft(ps)) * n;
% 
% %%% square wave
% signal = zeros(size(time));
% signal(sin(2*pi*3*time)>.9) = 1;
% 
% %%% AM (amplitude modulation)
% signal = 10*interp1(rand(1,10),linspace(1,10,n),'spline') .* sin(2*pi*40*time);
% 
% %%% FM (frequency modulation)
% freqmod = 20*interp1(rand(1,10),linspace(1,10,n));
% signal  = sin( 2*pi * ((10*time + cumsum(freqmod))/srate) );
% 
% %%% filtered noise
% signal = randn(size(time));
% s  = 5*(2*pi-1)/(4*pi);       % normalized width
% fx = exp(-.5*((hz-10)/s).^2); % gaussian
% fx = fx./max(fx);             % gain-normalize
% signal = 20*real( ifft( fft(signal).*fx) );



% compute amplitude spectrum
ampl = 2*abs(fft(signal)/n);

% plot in time domain
figure(1), clf
subplot(211)
plot(time,signal,'k','linew',2)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('Time domain')
set(gca,'xlim',[time(1)+.05 time(end)-.05],'ylim',[min(signal)*1.1 max(signal)*1.1])

% plot in frequency domain
subplot(212)
stem(hz,ampl,'k-s','linew',2,'markerfacecolor','k')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain')
set(gca,'xlim',[0 100])


%% 2D examples


%%% gabor patch
width = 20;   % width of gaussian
sphs  = pi/4; % sine phase

lims = [-91 91];
[x,y] = ndgrid(lims(1):lims(2),lims(1):lims(2));
xp = x*cos(sphs) + y*sin(sphs);
yp = y*cos(sphs) - x*sin(sphs);
gaus2d = exp(-(xp.^2 + yp.^2) ./ (2*width^2));
sine2d = sin( 2*pi*.02*xp );
img = sine2d .* gaus2d;


%%% white noise
img = randn(size(img));
% 
% 
% %%% portrait
% lenna = imread('Lenna.png');
% img   = double(mean(lenna,3));
% imgL  = img;
% 
% 
% %%% low-pass filtered Lenna
% width = .1;   % width of gaussian (normalized Z units)
% [x,y] = ndgrid(zscore(1:size(imgL,1)),zscore(1:size(imgL,2)));
% gaus2d= exp(-(x.^2 + y.^2) ./ (2*width^2)); % add 1- at beginning to invert filter
% imgX  = fftshift(fft2(imgL));
% img   = real(ifft2(fftshift(imgX.*gaus2d)));
% 
% 
% %%% high-pass filtered Lenna
% width = .3;   % width of gaussian (normalized Z units)
% [x,y] = ndgrid(zscore(1:size(imgL,1)),zscore(1:size(imgL,2)));
% gaus2d= 1-(exp(-(x.^2 + y.^2) ./ (2*width^2))); % add 1- at beginning to invert filter
% imgX  = fftshift(fft2(imgL));
% img   = real(ifft2(fftshift(imgX.*gaus2d)));
% 
% 
% 
% %%% phase-scrambled Lenna
% imgX  = fftshift(fft2(imgL));
% powr2 = abs(imgX);
% phas2 = angle(imgX);
% img   = real(ifft2(fftshift( powr2.*exp(1i*reshape(phas2(randperm(numel(phas2))),size(phas2))) )));





% power and phase spectra
imgX  = fftshift(fft2(img));
powr2 = log(abs(imgX));
phas2 = angle(imgX);


figure(2), clf
subplot(121)
imagesc(img), axis square
title('Space domain')

subplot(222)
imagesc(powr2), axis square
title('Amplitude spectrum')

subplot(224)
imagesc(phas2), axis square
title('Phase spectrum')

%% done.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
