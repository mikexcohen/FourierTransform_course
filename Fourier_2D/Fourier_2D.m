%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: 2D Fourier transform
%      VIDEO: How the 2D FFT works
% Instructor: mikexcohen.com
%
%%

%% movie to illustrate frequencies

% specify vector of sine frequencies
sinefreq = linspace(.0001,.2,50);  % arbitrary units


% leave this fixed for now
sinephas = pi/2;


% sine wave initializations
lims  = [-91 91];
[x,y] = ndgrid(lims(1):lims(2),lims(1):lims(2));
xp    = x*cos(sinephas) + y*sin(sinephas);
clim  = [0 1000];


% setup plot
figure(1), clf
subplot(121)
imageh = imagesc(rand(length(x)));
axis square, axis off, axis xy
title('Space domain')

subplot(222)
amph = imagesc(rand(length(x)));
axis square, axis off, axis xy
set(gca,'xlim',[lims(2)-30 lims(2)+30],'ylim',[lims(2)-30 lims(2)+30],'clim',clim)
title('Amplitude spectrum')

subplot(224)
phaseh = imagesc(rand(length(x)));
axis square, axis off, axis xy
set(gca,'xlim',[lims(2)-30 lims(2)+30],'ylim',[lims(2)-30 lims(2)+30])
title('Phase spectrum')


% now loop over phases
for si=1:length(sinefreq)
    
    % compute sine wave
    img = sin( 2*pi*sinefreq(si)*xp );
    
    % 2D FFT and extract power and phase spectra
    imgX  = fftshift(fft2(img));
    
    powr2 = abs(imgX);
    phas2 = angle(imgX);
    
    % update plots
    set(imageh,'CData',img);
    set(amph  ,'CData',powr2);
    set(phaseh,'CData',phas2);
    
    pause(.2)
end


%% movie to illustrate phases

% specify vector of sine phases
sinephas = linspace(0,pi,50);

% leave this fixed for now
sinefreq = .05;  % arbitrary units


% sine wave initializations
lims  = [-91 91];
[x,y] = ndgrid(lims(1):lims(2),lims(1):lims(2));
clim  = [0 1000];

% setup plot
figure(2), clf
subplot(121)
imageh = imagesc(rand(length(x)));
axis square, axis off, axis xy
title('Space domain')

subplot(222)
amph = imagesc(rand(length(x)));
axis square, axis off, axis xy
set(gca,'xlim',[lims(2)-30 lims(2)+30],'ylim',[lims(2)-30 lims(2)+30],'clim',clim)
title('Amplitude spectrum')

subplot(224)
phaseh = imagesc(rand(length(x)));
axis square, axis off, axis xy
set(gca,'xlim',[lims(2)-30 lims(2)+30],'ylim',[lims(2)-30 lims(2)+30])
title('Phase spectrum')


% now loop over phases
for si=1:length(sinephas)
    
    % x-coordinate values (based on rotating x and y)
    xp = x*cos(sinephas(si)) + y*sin(sinephas(si));
    
    % compute 2D sine patch
    img = sin( 2*pi*sinefreq*xp );
    
    % 2D FFT and extract power and phase spectra
    imgX  = fftshift(fft2(img));
    powr2 = abs(imgX);
    phas2 = angle(imgX);
    
    % update plots
    set(imageh,'CData',img);
    set(amph  ,'CData',powr2);
    set(phaseh,'CData',phas2);
    
    pause(.2)
end

%% physical location

lims  = [-91 91];
[x,y] = ndgrid(lims(1):lims(2),lims(1):lims(2));
width = 20;   % width of gaussian


centlocs = linspace(-80,80,50);


% setup plot
figure(3), clf
subplot(121)
imageh = imagesc(rand(length(x)));
axis square, axis off, axis xy
title('Space domain')

subplot(222)
amph = imagesc(rand(length(x)));
axis square, axis off, axis xy
title('Amplitude spectrum')
set(gca,'clim',[0 200])

subplot(224)
phaseh = imagesc(rand(length(x)));
set(gca,'xlim',[lims(2)-30 lims(2)+30],'ylim',[lims(2)-30 lims(2)+30])
axis square, axis off, axis xy
title('Phase spectrum')


% now loop over locations (center locations)
for si=1:length(centlocs)
    
    mx = x-centlocs(si);
    my = y-20;
    
    gaus2d = exp(-(mx.^2 + my.^2) ./ (2*width^2));
    img = zeros(size(gaus2d));
    img(gaus2d>.9) = 1;
    
    % 2D FFT and extract power and phase spectra
    imgX  = fftshift(fft2(img));
    powr2 = abs(imgX);
    phas2 = angle(imgX);
    
    % update plots
    set(imageh,'CData',img);
    set(amph  ,'CData',powr2);
    set(phaseh,'CData',phas2);
    
    pause(.2)
end

%% done.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
