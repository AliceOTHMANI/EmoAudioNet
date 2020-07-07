import os 
from PIL import Image

liste_dir = os.listdir()

x = 338
y = 144

x2 = 1710
y2 = 1069

for element in liste_dir : 
	if os.path.isdir(element):
		liste_files = os.listdir(element+'/')
		for img in liste_files :
			if not os.path.isdir(element+'/'+img):
				if element > str(439):
					image = Image.open(element+'/'+img)
					image = image.crop((x,y,x2,y2))
					image.save(element+'/'+img)
print("END")