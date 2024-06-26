# build_meva.py
import json
import os
from pathlib import Path
import pandas as pd
import datasets

class meva(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    DATASET_KEYS = ["A", "A'", "B", "B'", 'candidates', 'label', "A_str", "A'_str", "B_str", "B'_str"]
    HIDDEN_LABEL = '? (hidden)'
    QMARK_IMG = '/content/drive/MyDrive/meva/qmark.jpg'

    def _info(self):
        features = datasets.Features({
            "A": datasets.Image(),
            "A'": datasets.Image(),
            "B": datasets.Image(),
            "B'": datasets.Image(),
            "candidates_images": [datasets.Image()],
            "label": datasets.Value("int64"),
            "candidates": [datasets.Value("string")],
            "A_str": datasets.Value("string"),
            "A'_str": datasets.Value("string"),
            "B_str": datasets.Value("string"),
            "B'_str": datasets.Value("string"),
        })
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
    # Specify the paths to your CSV files and image folder in Colab
      examples_csv = '/content/drive/MyDrive/meva/meva_test.csv'
      images_dir = '/content/drive/MyDrive/meva/MeVA_images'

      # Create a dictionary with the paths
      data_dir = {
          'examples_csv': examples_csv,
          'images_dir': images_dir
    }

      # Create SplitGenerator instances for train, dev, and test splits
      #train_gen = datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=data_dir)
      #dev_gen = datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=data_dir)
      test_gen = datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=data_dir)

      return [test_gen]

# method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, examples_csv, images_dir):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
    
        df = pd.read_csv(examples_csv)
    
        for r_idx, r in df.iterrows():
            r_dict = r.to_dict()
            r_dict['candidates'] = json.loads(r_dict['candidates'])
            candidates_images = [os.path.join(images_dir, x) for x in r_dict['candidates']]
            r_dict['candidates_images'] = candidates_images
            r_dict["A_str"] = r_dict['A_img']
            r_dict["A'_str"] = r_dict['B_img']
            r_dict["B_str"] = r_dict['C_img']
            r_dict["B'_str"] = r_dict['D_img']
            for img in ['A_img', 'B_img', 'C_img', 'D_img']:
                if r_dict[img] == self.HIDDEN_LABEL:
                    r_dict[img] = os.path.join(images_dir, self.QMARK_IMG)
                else:
                    r_dict[img] = os.path.join(images_dir, r_dict[img])
            r_dict["A"] = r_dict['A_img']
            r_dict["A'"] = r_dict['B_img']
            r_dict["B"] = r_dict['C_img']
            r_dict["B'"] = r_dict['D_img']
            relevant_r_dict = {k:v for k,v in r_dict.items() if k in self.DATASET_KEYS or k == 'candidates_images'}
            yield r_idx, relevant_r_dict
