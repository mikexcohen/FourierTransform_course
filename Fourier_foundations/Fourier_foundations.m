%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Foundations
%      VIDEO: Complex numbers
% Instructor: mikexcohen.com
%
%%

% several ways to create a complex number
z = 4 + 3i;
z = 4 + 3*1i;
z = 4 + 3*sqrt(-1);
z = complex(4,3);

disp([ 'Real part is ' num2str(real(z)) ' and imaginary part is ' num2str(imag(z)) '.' ])


% beware of a common programming error:
i = 2;
zz = 4 + 3*i;


% plot the complex number
figure(1), clf
plot(real(z),imag(z),'s','markersize',12,'markerfacecolor','k')

% make plot look nicer
set(gca,'xlim',[-5 5],'ylim',[-5 5])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Real axis')
ylabel('Imaginary axis')
title([ 'Number (' num2str(real(z)) ' ' num2str(imag(z)) 'i) on the complex plane' ])

%% compute the two key properties of a complex number

% magnitude of the number (distance to origin)
magZ = sqrt( real(z)^2 + imag(z)^2 );
magZ = abs( z );

% angle of the line relative to positive real axis
angZ = atan2( imag(z),real(z) );
angZ = angle( z );

%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Foundations
%      VIDEO: Euler's formula
% Instructor: mikexcohen.com
%
%%

% plot e^x
figure(2), clf
x = linspace(-3,3,50);
plot(x,exp(x),'k','linew',3)
axis square
xlabel('x')
legend({'f(x) = e^x'})
grid minor

%% [cos(k) sin(k)] are on the unit circle for any k

% any real number k
k = -654321657546546;

% and plot
figure(3), clf
h1 = plot( cos(k),sin(k) ,'ko','markerfacecolor','r','markersize',15);

% make plot look nicer
set(gca,'xlim',[-1.5 1.5],'ylim',[-1.5 1.5])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Cosine axis')
ylabel('Sine axis')

% also draw a unit circle
x = linspace(-pi,pi,100);
h = plot(cos(x),sin(x));
set(h,'color',[1 1 1]*.7) % light gray

uistack(h1,'top'); % put red circle on top 

%% ... and so is the angle defined by Euler's formula

euler = exp( 1i*k );

% draw a line using polar notation
h = polar([0 angle(euler)],[0 1],'r');
set(h,'linewidth',2)

%% Euler's formula with arbitrary vector magnitude

% use Euler's formula to plot vectors
m = 4;
k = pi/3;
compnum = m*exp( 1i*k );

% extract magnitude and angle
mag = abs(compnum);
phs = angle(compnum);

figure(4), clf
plot( real(compnum) , imag(compnum) ,'ro','linew',2,'markersize',10,'markerfacecolor','r')

% make plot look nicer
set(gca,'xlim',[-5 5],'ylim',[-5 5])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Real axis')
ylabel('Imaginary axis')

% draw a line using polar notation
h = polar([0 phs],[0 mag],'r');
set(h,'linewidth',2)

% title 
title([ 'Rectangular: [' num2str(real(compnum)) ' ' num2str(imag(compnum)) 'i ], ' ...
        'Polar: ' num2str(mag) 'e^{i' num2str(phs) '}' ])
    
%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Foundations
%      VIDEO: Sine waves and complex sine waves
% Instructor: mikexcohen.com
%
%%


%% sine wave parameters

% sine wave parameters
freq = 3;    % frequency in Hz
ampl = 2;    % amplitude in a.u.
phas = pi/3; % phase in radians

% general simulation parameters
srate = 500; % sampling rate in Hz
time  = 0:1/srate:2-1/srate; % time in seconds

% generate the sine wave
sinewave = ampl * sin( 2*pi * freq * time + phas );

figure(1), clf
plot(time,sinewave,'ks-','linewidth',2,'markerfacecolor','w')
xlabel('Time (sec.)'), ylabel('Amplitude')
title('A sine wave.')

%% sine and cosine are the same but for a phase shift

% generate the sine wave
sinewave = ampl * sin( 2*pi * freq * time + phas );
coswave  = ampl * cos( 2*pi * freq * time + phas );

figure(2), clf
plot(time,sinewave,'k','linewidth',2)
hold on
plot(time,coswave,'r','linewidth',2)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('A sine and cosine with the same parameters.')

%% little GUI

% a little interactive GUI to give you a better sense of sine wave parameters
sinewave_from_params;

% Note that this GUI doesn't work in Octave.

%% complex sine waves

% general simulation parameters
srate = 500; % sampling rate in Hz
time  = 0:1/srate:2-1/srate; % time in seconds

% sine wave parameters
freq = 5;    % frequency in Hz
ampl = 2;    % amplitude in a.u.
phas = 0*pi/3; % phase in radians

% generate the sine wave
csw = ampl * exp( 1i* (2*pi * freq * time + phas) );

figure(3), clf
subplot(211)
plot(time,real(csw), time,imag(csw),'linew',2)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('Complex sine wave projections')
legend({'real';'imag'})

subplot(212)
plot3(time,real(csw),imag(csw),'k','linew',3)
xlabel('Time (sec.)'), ylabel('real part'), zlabel('imag. part')
set(gca,'ylim',[-1 1]*ampl*3,'zlim',[-1 1]*ampl*3)
axis square
rotate3d on

%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Foundations
%      VIDEO: The dot product
% Instructor: mikexcohen.com
%
%%

% two vectors
v1 = [ 1 2 3 ];
v2 = [ 3 2 1 ];

% compute the dot product
dp = sum( v1.*v2 );

%% dot products of sine waves

% general simulation parameters
srate = 500; % sampling rate in Hz
time  = 0:1/srate:2-1/srate; % time in seconds

% sine wave parameters
freq1 = 5;    % frequency in Hz
freq2 = 5;    % frequency in Hz

ampl1 = 2;    % amplitude in a.u.
ampl2 = 2;    % amplitude in a.u.

phas1 = 0; % phase in radians
phas2 = pi/2; % phase in radians

% generate the sine wave
sinewave1 = ampl1 * sin( 2*pi * freq1 * time + phas1 );
sinewave2 = ampl2 * sin( 2*pi * freq2 * time + phas2 );

% compute dot product
dp = dot(sinewave1,sinewave2);

% show result in command window
disp([ 'dp = ' num2str( dp ) ])

%% with a signal

% phase of signal
theta = 0*pi/4;


% simulation parameters
srate = 1000;
time  = -1:1/srate:1;

% signal
signal = sin(2*pi*5*time + theta) .* exp( (-time.^2) / .1);

% sine wave frequencies
sinefrex = 2:.5:10;

% plot signal
figure(3), clf
subplot(211)
plot(time,signal,'k','linew',3)
xlabel('Time (sec.)'), ylabel('Amplitude (a.u.)')
title('Signal')

dps = zeros(size(sinefrex));
for fi=1:length(dps)
    
    % create sine wave
    sinew = sin( 2*pi*sinefrex(fi)*time);
    
    % compute dot product
    dps(fi) = dot( sinew,signal ) / length(time);
end

% and plot
subplot(212)
stem(sinefrex,dps,'k','linew',3,'markersize',10,'markerfacecolor','w')
set(gca,'xlim',[sinefrex(1)-.5 sinefrex(end)+.5],'ylim',[-.2 .2])
xlabel('Sine wave frequency (Hz)')
ylabel('Dot product (signed magnitude)')
title([ 'Dot product of signal and sine waves (' num2str(theta) ' rad. offset)' ])

%%
%     COURSE: Understand the Fourier transform and its applications
%    SECTION: Foundations
%      VIDEO: Complex dot product
% Instructor: mikexcohen.com
%
%%


%% Same as previous video but with complex sine wave

% phase of signal
theta = 0*pi/4;


% simulation parameters
srate = 1000;
time  = -1:1/srate:1;

% signal
signal = sin(2*pi*5*time + theta) .* exp( (-time.^2) / .1);

% sine wave frequencies
sinefrex = 2:.5:10;

% plot signal
figure(3), clf
subplot(211)
plot(time,signal,'k','linew',3)
xlabel('Time (sec.)'), ylabel('Amplitude (a.u.)')
title('Signal')

dps = zeros(size(sinefrex));
for fi=1:length(dps)
    
    % create complex sine wave
    sinew = exp( 1i*2*pi*sinefrex(fi)*time);
    
    % compute dot product
    dps(fi) = dot( sinew,signal ) / length(time);
end

% and plot
subplot(212)
stem(sinefrex,abs(dps),'k','linew',3,'markersize',10,'markerfacecolor','w')
set(gca,'xlim',[sinefrex(1)-.5 sinefrex(end)+.5],'ylim',[-.2 .2])
xlabel('Sine wave frequency (Hz)')
ylabel('Dot product (unsigned magnitude)')
title([ 'Dot product of signal and sine waves (' num2str(theta) ' rad. offset)' ])

%% more detail...

% phase of signal
theta = 0*pi/4;

% signal
signal = sin(2*pi*5*time + theta) .* exp( (-time.^2) / .1);

% create sine and cosine waves
sinew = sin( 2*pi*5*time );
cosnw = cos( 2*pi*5*time );

% compute dot products
dps = dot( sinew,signal ) / length(time);
dpc = dot( cosnw,signal ) / length(time);

% combine sine and cosine into complex dot product
dp_complex = complex(dpc,dps); % cos/sin were swapped in the video
mag = abs(dp_complex);
phs = angle(dp_complex);

% and plot
figure(4), clf
plot( dpc , dps ,'ro','linew',2,'markersize',10,'markerfacecolor','r')

% make plot look nicer
set(gca,'xlim',[-.2 .2],'ylim',[-.2 .2])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Cosine axis')
ylabel('Sine axis')

% draw a line using polar notation
h = polar([0 phs],[0 mag],'r');
set(h,'linewidth',2)

% title 
title([ 'Rectangular: [' num2str(real(dp_complex)) ' ' num2str(imag(dp_complex)) 'i ], ' ...
        'Polar: ' num2str(mag) 'e^{i' num2str(phs) '}' ])

%% illustration of the effect of phase offsets on dot products

% create complex sine wave
csw = exp( 1i*2*pi*5*time );
rsw = cos(    2*pi*5*time );

% specify range of phase offsets for signal
phases = linspace(0,7*pi/2,100);


% setup the plot
figure(5), clf
subplot(223)
ch = plot(0,0,'ro','linew',2,'markersize',10,'markerfacecolor','r');
set(gca,'xlim',[-.2 .2],'ylim',[-.2 .2])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Cosine axis')
ylabel('Sine axis')
title('Complex plane')

% and then setup the plot for the real-dot product axis
subplot(224)
rh = plot(0,0,'ro','linew',2,'markersize',10,'markerfacecolor','r');
set(gca,'xlim',[-.2 .2],'ylim',[-.2 .2],'ytick',[])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Real axis')
title('Real number line')


for phi=1:length(phases)
    
    % create signal
    signal = sin(2*pi*5*time + phases(phi)) .* exp( (-time.^2) / .1);
    
    % compute complex dot product
    cdp = sum( signal.*csw ) / length(time);
    
    % compute real-valued dot product
    rdp = sum( signal.*rsw ) / length(time);
    
    % plot signal and real part of sine wave
    subplot(211)
    plot(time,signal, time,rsw,'linew',2)
    title('Signal and sine wave over time')
    
    % plot complex dot product
    subplot(223)
    set(ch,'XData',real(cdp),'YData',imag(cdp))
    
    % draw normal dot product
    subplot(224)
    set(rh,'XData',rdp)
    
    % wait a bit
    pause(.1)
end

%% end.

% Interested in more courses? See sincxpress.com 
% Use code MXC-DISC4ALL for the lowest price for all courses.
