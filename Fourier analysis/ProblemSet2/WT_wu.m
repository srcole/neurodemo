function coef = WT_wu(sig, samp_rate, f, t, is_plot)
% Function modified from: http://sccn.ucsd.edu/pipermail/eeglablist/2003/000240.html
% Original author: Wu Xaing
%Usage
%   coef = TFWT_wu(sig, samp_rate, is_plot)
%Inupts
%   sig             1-d signal
%   samp_rate       sampling rate
%   f               2-element array [lo hi] frequency limits in Hz
%   t               time array of the data (used for plotting)
%   is_plot         1,plot;0,not plot
%Outputs
%   coefficients    time_frequency distribution.
%Algorithm
%   The signal was convoluted by complex Morlet's wavelets w(t,f0) having
%a Gaussian shape both in the time domain(SD_t) and in the frequency
%domain(SD_f) around its central frequency f0:w(t,f0) =
%(SD_t*sqrt(pi))^(-1/2) * exp( -t.^2/(2*SD_t^2) ) .* exp(i*2*pi*f0*t),
%with SD_f = 1/(2*pi*SD_t).f0/SD_f = 7,with f0 ranging from 20 to 100 Hz
%in 1 Hz step.At 20 Hz,this leads to a vavelet duration (2*SD_t) of 111.4ms
%and to a spectral bandwidth (2*SD_f) of 5.8 Hz and at 100 Hz, to a duration
%of 22.2 ms and a bandwidth of 28.6 Hz.The time resolution of this method,
%therefore, increase with frequency,whereas the frequency resolution decreases.
sig = sig(:)';
len_sig = length(sig);
samp_period = 1/samp_rate;
%initialize output coefficients matrix
row_coef = f(2)-f(1)+1;
col_coef = len_sig;
coef = zeros(row_coef,col_coef);
%compute coefficients
for freq = f(1):f(2)
    SD_f = freq/7;
    SD_t = 1/(2*pi*SD_f);
    x = -SD_t:samp_period:SD_t;
    Morlets = ( SD_t*sqrt(pi) )^(-1/2) * exp( -x.^2/(2*SD_t^2) ) .* exp(i*2*pi*freq*x );
    coef_freq = conv(sig,Morlets);
    coef(freq-f(1)+1,:) = coef_freq(round(length(x)/2):col_coef+round(length(x)/2)-1 );
end

%plot
colormap('gray');
if is_plot==1
    imagesc(t, [f(1):f(2)],abs(coef).^2);
    xlabel('time (second)');
    ylabel('frequency (Hz)');
    axis('xy');
    c = colorbar;
    c.Label.String = 'Amplitude (a.u.)';
    shading flat;
    zoom on;
end
