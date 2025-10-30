import os
import pandas as pd
from tqdm import tqdm

img_directory = "../data/utkface_aligned_cropped/UTKFace"
data = []

print(f'scanning file:{img_directory}')
for filename in tqdm(os.listdir(img_directory)):
    if filename.endswith(".jpg"):
        filepath = os.path.join(img_directory, filename)

        try:
            parts = filename.split("_") # file format [age]_[gender]_[race]_[date&time].jpg

            #making sure at least 4 parts of img
            if len(parts) >= 4:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                data.append({
                    'filepath':filepath,
                    'age':age,
                    'gender':gender,
                    'race':race,
                    'filename':filename
                })
            else:
                print(f'file format incorrect, skipped{filename}')

        except ValueError:
            print(f'parsing failed, skipped{filename}')#[cite:50]
        except Exception as e:
            print(f'unkown failure reason{e}, skipped{filename}')


df = pd.DataFrame(data)

#map 0 and 1 to words
gender_map = {0:'male', 1:'female'}
race_map = {
    0:'white',
    1:'black',
    2:'asian',
    3:'indian',
    4:'other'
}
df['gender_label'] = df['gender'].map(gender_map)
df['race_label'] = df['race'].map(race_map)

out_csv_path = "../data/data.csv"
df.to_csv(out_csv_path, index=False)

print(f'1-5th rows:')
print(df.head(5))