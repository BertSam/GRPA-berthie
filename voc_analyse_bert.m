clear all;
close all;
clc;

%% Chargement du signal
cd('/home/berthie/Documents/t5/GRPA-berthie')
[audioIn,fs] = audioread('NO_S.wav');

ts = 1/fs;

temp_des_seq = 10/1000; % dur�e d�sir�e de la s�quence

audioIn = audioIn(1:temp_des_seq*fs); % Garder juste une seconde, pour aller plus vite

audioIn = audioIn-mean(audioIn);

audioIn = audioIn./max(abs(audioIn));

% Afficher le signal pour voir ce dontz il s'agit
t = (0:length(audioIn)-1)/fs;
% figure(1)
% 
% plot(t,audioIn);
% legend('Signal source');
% xlabel('Temps (s)')

% Analyse LPC pour chacune des fen�tres

ordre_lpc = 5;

Az = lpc(audioIn,ordre_lpc);
% freqz(Az,1)

lsp = poly2lsf(Az);

%trames_residu = zeros(numel(audioIn), 1);

trames_residu = filter([0, -Az(2:end)],1,audioIn);

res = audioIn-trames_residu;

res_test = filter(Az,1,audioIn);



figure
plot(t,audioIn);
hold on 
plot(t,trames_residu);
plot(t,res)
plot(t,res_test+0.1)
legend('Signal source','Pr�diction','R�sidu de pr�diction','R�sidu de pr�diction test');
xlabel('Temps (s)')

%% Analyse LPC

% D�composer audioIn en fen�tres de 30ms avec des sauts de 10ms

l_wind = round(fs*0.03);
l_step = round(fs*0.010);
nb_win = floor((length(audioIn)-(l_wind-l_step))/l_step);
winds = zeros(l_wind, nb_win);
for i = 0:nb_win-1
    winds(:,i+1) = audioIn(i*l_step+1:i*l_step+l_wind);
end

% Analyse LPC pour chacune des fen�tres

ordre_lpc = 16;
Az = lpc(winds,ordre_lpc);

% D�composer audioIn en trames de 10ms (on saute le 1er 10ms pour que le
% premier filtre LPC corresponde � la premi�re trame)

tp = (l_step+1:nb_win*l_step+l_step)/fs;
trames = reshape(audioIn(l_step+1:nb_win*l_step+l_step),[l_step, nb_win]);

% Calcul du r�sidu de pr�diction

trames_residu = zeros(l_step, nb_win);
memoire = zeros(1, ordre_lpc);
for tt = 1:nb_win
    [trames_residu(:,tt),memoire] = filter(Az(tt,:),1,trames(:,tt),memoire);
end

residu=reshape(trames_residu,l_step*nb_win,1);

% Affichage du r�sidu

figure(2);
hold;
plot(tp,audioIn(l_step+1:l_step*nb_win+l_step),'b');
plot(tp,residu,'r');
legend('Signal source','R�sidu de pr�diction');
xlabel('Temps (s)')

% Re-synth�se du signal

trames_synthese = zeros(l_step,nb_win);
memoire_s = zeros(1,ordre_lpc);
for tt = 1:nb_win
    [trames_synthese(:,tt),memoire_s] = filter(1,Az(tt,:),trames_residu(:,tt),memoire_s);
end

synthese=reshape(trames_synthese,l_step*nb_win,1);

% Affichage du signal resynth�tis�
figure(3);
hold;
plot(tp,audioIn(l_step+1:l_step*nb_win+l_step),'b');
plot(tp,synthese,'r');
legend('Signal source','Signal resynth�tis�');
xlabel('Temps (s)')

% % Jouer les signaux (notre oreille est plus sensible que nos yeux �
% % certains d�fauts sonores, qu'on appelle souvent artefacts de traitement)
% 
% disp('Appuyer sur une touche pour jouer le signal d''origine')
% pause;
% sound(audioIn, fs);
% disp('Appuyer sur une touche pour jouer le residu de prediction')
% pause;
% sound(residu, fs);
% disp('Appuyer sur une touche pour jouer le signal de synthese')
% pause;
% sound(synthese, fs);

%% Calcul de l'�nergie (serait plus lisse si calcul�e pitch-synchrone)

ener=zeros(nb_win,1);
for tt = 1:nb_win
    ener(tt)=trames_residu(:,tt)'*trames_residu(:,tt);
end

te=(l_step:l_step:nb_win*l_step)/fs;
figure(4);
subplot(2,1,1); plot(t,audioIn);
subplot(2,1,2); plot(te,10.0*log10(ener));
xlabel('Temps (s)');
ylabel('Enegie (dB)');

%% Analyse de pitch (TODO: SYNCHRONISER AVEC LPC ET ENERGIE)

% Use the 'NCF' method and specify an 60 ms window length with a 
% 10 ms hop. Limit the search range to 60-250 Hz and postprocess the
% pitch contour with a 3-element median filter. Plot the results.

[f0,idx] = pitch(audioIn,fs, ...
    'Method','NCF', ...
    'WindowLength',round(fs*0.06), ...
    'OverlapLength',round(fs*(0.06-0.01)), ...
    'Range',[60,250], ...
    'MedianFilterLength',3);

t = (0:length(audioIn)-1)/fs;
t0 = (idx - 1)/fs;
figure(5);
subplot(2,1,1); plot(t,audioIn);
subplot(2,1,2); plot(t0,f0);
xlabel('Temps (s)')
ylabel('Pitch (Hz)')
ylim([50 300])
