function varargout = sinewave_from_params(varargin)
% SINEWAVE_FROM_PARAMS MATLAB code for sinewave_from_params.fig
%      SINEWAVE_FROM_PARAMS, by itself, creates a new SINEWAVE_FROM_PARAMS or raises the existing
%      singleton*.
%
%      H = SINEWAVE_FROM_PARAMS returns the handle to a new SINEWAVE_FROM_PARAMS or the handle to
%      the existing singleton*.
%
%      SINEWAVE_FROM_PARAMS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SINEWAVE_FROM_PARAMS.M with the given input arguments.
%
%      SINEWAVE_FROM_PARAMS('Property','Value',...) creates a new SINEWAVE_FROM_PARAMS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before sinewave_from_params_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to sinewave_from_params_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help sinewave_from_params

% Last Modified by GUIDE v2.5 14-Nov-2017 15:45:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @sinewave_from_params_OpeningFcn, ...
                   'gui_OutputFcn',  @sinewave_from_params_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT





function sinewave_from_params_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to sinewave_from_params (see VARARGIN)

% Choose default command line output for sinewave_from_params
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
cla(handles.polarax)
hold(handles.polarax,'on')
h(1) = plot(handles.polarax,[-2 2],[0 0],'k','linew',2);
h(2) = plot(handles.polarax,[0 0],[-2 2],'k','linew',2);
h(3) = plot(handles.polarax,sin(linspace(-pi,pi,30)),cos(linspace(-pi,pi,30)),'k--');
h(4) = plot(handles.polarax,2*sin(linspace(-pi,pi,30)),2*cos(linspace(-pi,pi,30)),'k:');
set(h,'HitTest','off')

set(handles.polarax,'xlim',[-2 2],'ylim',[-2 2])
axis(handles.polarax,'square')
xlabel(handles.polarax,'real')
ylabel(handles.polarax,'imag')

% UIWAIT makes sinewave_from_params wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = sinewave_from_params_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on mouse press over axes background.
function polarax_ButtonDownFcn(hObject, eventdata, handles)
updategraphs(handles)

function updategraphs(handles)
% get mouse click point
xy = get(handles.polarax,'CurrentPoint');
xy = xy(1,1:2);

% convert to polar coords
amp = sqrt(sum(xy.^2));
phs = atan2(xy(2),xy(1));

% clear existing line from axis
h = get(handles.polarax,'Children');
for i=1:length(h)
    if isequal(get(h(i),'Color'),[1 0 0])
        delete(h(i));
    end
end

% draw a vector on the plot
h1(1) = plot(handles.polarax,xy(1),xy(2),'ro','markerfacecolor','w','markersize',10);
h1(2) = plot(handles.polarax,[0 xy(1)],[0 xy(2)],'r');
set(h1,'HitTest','off')
title([ 'Amplitude: ' num2str(round(amp*100)/100) ', phase: ' num2str(round(phs*100)/100) ' rad.' ])

% update slider display
hz = get(handles.freqslider,'Value');
hz = hz*8+2;
set(handles.text2,'String',[ 'Frequency: ' num2str(hz) ' Hz' ])

fs = 1000;
tm = -1:1/fs:2-1;
sw = amp * sin(2*pi*hz*tm + phs);
hold(handles.timeax,'off')
plot(handles.timeax,tm,sw,'k')
set(handles.timeax,'ylim',[-3 3])
hold(handles.timeax,'on')
plot(handles.timeax,tm([1 end]),[-1 -1],'k--')
plot(handles.timeax,tm([1 end]),[1 1],'k--')
plot(handles.timeax,tm([1 end]),[-2 -2],'k:')
plot(handles.timeax,tm([1 end]),[2 2],'k:')
plot(handles.timeax,[0 0],get(handles.timeax,'ylim'),'k')
xlabel(handles.timeax,'Time (sec.)')


% --- Executes on slider movement.
function freqslider_Callback(hObject, eventdata, handles)
updategraphs(handles)


% --- Executes during object creation, after setting all properties.
function freqslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freqslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
