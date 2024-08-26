import os 
import glob


translate_map = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
for folder in glob.glob(r"C:\Users\ayhan\Desktop\ml-collection\data\ten_animals\raw-img\*"):
    # rename the folder
    folder_name = os.path.basename(folder)
    try:
        new_folder_name = translate_map[folder_name]
    except KeyError:
        for k,v in translate_map.items():
            if folder_name == v:
                new_folder_name = k
    os.rename(folder, os.path.join(os.path.dirname(folder), new_folder_name))
