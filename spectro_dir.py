import os
import wave
import pylab
import pandas as pd
from PIL import Image
from resizeimage import resizeimage
import librosa
from nlpaug.util.visual.wave import VisualWave
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
import scipy.io.wavfile as wavf


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

### FONCTION CREATION SPECTROGRAMME ###

def graph_spectrogram(wav_file, name, path):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(path+name)
    


def create_spectro_dir(delete):
    """ Cette fonction permet de créer tous les spectrogrammes correspondants aux fichiers .wav présents dans l'arborescence
        delete : boolean indiquant si l'on supprimer ou non les fichiers .wav
    """
    for dossier in os.listdir('.') : # on parcourt tous les dossiers du répertoire courant
        if os.path.isdir(dossier): # si c'est bien un dossier 
            cpt = 0
            for wav in os.listdir('./'+dossier+'/') : # on parcours tous les fichiers .wav de ce repertoire
                if not os.path.isdir(wav) : # si c'est bien un fichier .wav
                    print("Spectrogramme du fichier :",'./'+dossier+'/'+wav,"crée") # trace d'exécution
                    graph_spectrogram('./'+dossier+'/'+wav, str(cpt) + '.png', './'+dossier+'/') # création du spectrogramme
                    
                    if delete : # on choisit de supprimer ou non les fichiers .wav
                        os.remove('./'+dossier+'/'+wav) # supprime le fichier .wav
                    cpt += 1

    print("FIN : Les fichiers sont tous traités")


def trouve_labels(num_dossier, csv_file):
	df = pd.read_csv(csv_file)
	matrice = df.as_matrix()

	arousal = 0
	valence = 0

	for ligne in matrice : 
		if str(ligne[0]) == num_dossier :
			arousal = ligne[1]
			valence = ligne[2]
			break

	return arousal, valence

def compte_data():
    """ Cette fonction permet de compter le nombre de données qu'il y'a dans l'arborescence
    """
    cpt = 0
    for dossier in os.listdir('.') : # on parcourt tous les dossiers du répertoire courant
        if os.path.isdir(dossier): # si c'est bien un dossier 
            for file in os.listdir('./'+dossier+'/') : # on parcours tous les fichiers .wav de ce repertoire
                if not os.path.isdir(file) : # si c'est bien un fichier .wav
                    cpt += 1

    print("Il y'a ::",cpt,"données")


def create_csv(csv_file):
	"""
	cette fonction permet de créer une nouveau fichier .csv avec tous les labels correspondants (ici avec 1308 lignes)
	à partir d'un fichier .csv plus compacte
	"""

	csv = []
	colonnes = ['fichier', 'arousal', 'valence']
	for dossier in os.listdir('.') : # on parcourt tous les dossiers du répertoire courant
		if os.path.isdir(dossier): # si c'est bien un dossier 
			arousal, valence = trouve_labels(dossier, csv_file) # à changer
			for file in os.listdir('./'+dossier+'/') : # on parcours tous les fichiers .wav de ce repertoire
				if not os.path.isdir(file) : # si c'est bien un fichier .wav
					ligne = ['DATA_AUDIO_AUG/'+dossier+'/'+file, arousal, valence]
					csv.append(ligne)

	df = pd.DataFrame(csv, columns = colonnes)

	df.to_csv('labels_audio.csv', encoding='utf-8')




def resize(x, y) :
	"""
	redimmensionne toutes les images contenues dans les dossiers avec les tailles x et y
	""" 

	for dossier in os.listdir('.') : # on parcourt tous les dossiers du répertoire courant
		if os.path.isdir(dossier): # si c'est bien un dossier 
			arousal, valence = trouve_labels(dossier, csv_file) # à changer
			for file in os.listdir('./'+dossier+'/') : # on parcours tous les fichiers .wav de ce repertoire
				if not os.path.isdir(file) : # si c'est bien un fichier .wav
					with open(file, 'r+b') as f :
						with Image.open(f) as image : 
							cover = resizeimage.resize_cover(image, [x,y])
							cover.save(file, image.format)




def augmentation() :
	doss = 0
	for dossier in os.listdir('.') : # on parcourt tous les dossiers du répertoire courant
		if os.path.isdir(dossier): # si c'est bien un dossier 
			cpt = 0
			doss += 1
			print("Dossier "+str(doss)+" sur 28")
			if doss >= 14 :
				for file in os.listdir('./'+dossier+'/') : # on parcours tous les fichiers .wav de ce repertoire
					if not os.path.isdir(file) : # si c'est bien un fichier .wav
						path = './'+dossier+'/'+file
						audio, sampling_rate = librosa.load(path)
						freq = sampling_rate
						"""
						Noise injection
						It simply add some random value into data.
						"""
						liste_factor = [0.03, 0.02, 0.01]

						for fact in liste_factor :
							cpt += 1
							aug = naa.NoiseAug(nosie_factor=fact)
							augmented_audio = aug.substitute(audio)
							path = './'+dossier+'/'+str(cpt)+'_'+file
							out_f = path
							wavf.write(out_f, freq, augmented_audio)

						liste_factor = [0.5, 2, 5]

						for fact in liste_factor :
							cpt += 1
							aug = naa.PitchAug(sampling_rate=sampling_rate, pitch_factor=fact)
							augmented_audio = aug.substitute(audio)
							path = './'+dossier+'/'+str(cpt)+'_'+file
							out_f = path
							wavf.write(out_f, freq, augmented_audio)



def renommage():
	"""
	renomme tous les fichiers .wav
	"""
	cpt = 0
	for dossier in os.listdir('.') : # on parcourt tous les dossiers du répertoire courant
		if os.path.isdir(dossier): # si c'est bien un dossier 
			for file in os.listdir('./'+dossier+'/') : # on parcours tous les fichiers .wav de ce repertoire
				if not os.path.isdir(file) : # si c'est bien un fichier .wav
					os.rename('./'+dossier+'/'+file,'./'+dossier+'/'+str(cpt)+".wav")
					cpt += 1




#create_csv('labels.csv')

#DELETE = True
#create_spectro_dir(DELETE)
compte_data()
#renommage()
#df = pd.read_csv('labels_spectro.csv')
#matrice = df.as_matrix()
#print(df)

