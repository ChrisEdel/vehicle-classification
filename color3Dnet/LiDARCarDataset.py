import tensorflow_datasets as tfds
import os
import sys
import numpy as np
import random
import tensorflow as tf
import re

DATE_PATTERN = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}')

DEFAULT_LABEL = 'unlabeled'

CAR = ['no', 'yes', DEFAULT_LABEL]

MANUFACTURERS = ['alfa-romeo',
                 'audi',
                 'bmw',
                 'citroen',
                 'dacia',
                 'dodge',
                 'fiat',
                 'ford',
                 'hyundai',
                 'iveco',
                 'jaguar',
                 'jeep',
                 'kia',
                 'lotus',
                 'maserati',
                 'mazda',
                 'mercedes',
                 'mini',
                 'nissan',
                 'opel',
                 'peugeot',
                 'porsche',
                 'range-rover',
                 'renault',
                 'seat',
                 'skoda',
                 'subaru',
                 'toyota',
                 'volvo',
                 'vw',
                 DEFAULT_LABEL]

MANUFACTURERS_MAIN = ['audi',
                      'volvo',
                      'ford',
                      'mercedes',
                      'vw']

MANUFACTURERS_FORD_VW = ['ford',
                         'vw']

MANUFACTURERS_AUDI_MERCEDES = ['audi',
                               'mercedes']

TYPES = ['alfa-romeo_4c',
         'alfa-romeo_stelvio',
         'audi_a3',
         'audi_a4',
         'audi_a5',
         'audi_a6',
         'audi_e-tron',
         'audi_q2',
         'audi_q3',
         'audi_q5',
         'audi_q7',
         'audi_rs4',
         'audi_s4',
         'audi_s5',
         'audi_unlabeled',
         'bmw_123d',
         'bmw_218d',
         'bmw_320d',
         'bmw_520d',
         'bmw_530d',
         'bmw_530e',
         'bmw_635',
         'bmw_640i',
         'bmw_m4',
         'bmw_unlabeled',
         'citroen_berlingo',
         'citroen_c3',
         'citroen_c4',
         'citroen_c5',
         'citroen_jumper',
         'citroen_spacetourer',
         'dacia_sandero',
         'dodge_ram',
         'fiat_bravo',
         'fiat_tipo',
         'ford_c-max',
         'ford_ecosport',
         'ford_edge',
         'ford_explorer',
         'ford_fiesta',
         'ford_focus',
         'ford_galaxy',
         'ford_kuga',
         'ford_mondeo',
         'ford_mondero',
         'ford_puma',
         'ford_ranger',
         'ford_s-max',
         'ford_tourneo',
         'ford_tournero',
         'ford_transit',
         'ford_turneo',
         'ford_unlabeled',
         'ford_vignale',
         'hyundai_ioniq',
         'iveco_unlabeled',
         'jaguar_f-type',
         'jeep_grand',
         'kia_carens',
         'kia_ceed',
         'kia_sportage',
         'lotus_elise',
         'maserati_grantourismo',
         'mazda_3',
         'mazda_cx-5',
         'mazda_mazda2',
         'mazda_mazda3',
         'mercedes_a180d',
         'mercedes_b180d',
         'mercedes_b200d',
         'mercedes_b250e',
         'mercedes_c180dt',
         'mercedes_c180t',
         'mercedes_c200d',
         'mercedes_c200dt',
         'mercedes_c220dt',
         'mercedes_cl',
         'mercedes_cla180',
         'mercedes_cla220d',
         'mercedes_cls',
         'mercedes_e',
         'mercedes_e200d',
         'mercedes_e220d',
         'mercedes_e280',
         'mercedes_gla200d',
         'mercedes_glc220d',
         'mercedes_glc300',
         'mercedes_sprinter',
         'mini_cooper',
         'mini_hatch',
         'nissan_qashqai',
         'opel_astra',
         'peugeot_2008',
         'peugeot_208',
         'peugeot_3008',
         'peugeot_308',
         'peugeot_5008',
         'peugeot_508sw',
         'peugeot_e-2008',
         'peugeot_rifter',
         'porsche_912',
         'porsche_964',
         'porsche_carrera',
         'porsche_cayenne',
         'porsche_macan',
         'range-rover_velar',
         'renault_clio',
         'renault_twingo',
         'seat_alhambra',
         'seat_arona',
         'seat_ateca',
         'seat_ibiza',
         'seat_laguna',
         'seat_leon',
         'seat_mii',
         'skoda_fabia',
         'skoda_oktavia',
         'skoda_rapid-spaceback',
         'subaru_xv',
         'toyota_auris',
         'volvo_s60',
         'volvo_s90',
         'volvo_v40',
         'volvo_v60',
         'volvo_v90',
         'volvo_xc40',
         'volvo_xc60',
         'volvo_xc90',
         'vw_aventura',
         'vw_california',
         'vw_golf',
         'vw_id.3',
         'vw_multivan',
         'vw_passat',
         'vw_polo',
         'vw_scirocco',
         'vw_sharan',
         'vw_t-cross',
         'vw_t-roc',
         'vw_t6',
         'vw_tiguan',
         'vw_touran',
         'vw_transporter',
         'vw_unlabeled',
         DEFAULT_LABEL]

TYPES_AUDI = ['audi_a3',
              'audi_a4',
              'audi_a5',
              'audi_a6',
              'audi_e-tron',
              'audi_q2',
              'audi_q3',
              'audi_q5',
              'audi_q7',
              'audi_rs4',
              'audi_s4',
              'audi_s5']

TYPES_VOLVO = ['volvo_s60',
               'volvo_s90',
               'volvo_v40',
               'volvo_v60',
               'volvo_v90',
               'volvo_xc40',
               'volvo_xc60',
               'volvo_xc90']

TYPES_FORD = ['ford_c-max',
              'ford_ecosport',
              'ford_edge',
              'ford_explorer',
              'ford_fiesta',
              'ford_focus',
              'ford_galaxy',
              'ford_kuga',
              'ford_mondeo',
              'ford_mondero',
              'ford_puma',
              'ford_ranger',
              'ford_s-max',
              'ford_tourneo',
              'ford_tournero',
              'ford_transit',
              'ford_turneo',
              'ford_unlabeled',
              'ford_vignale']

TYPES_MERCEDES = ['mercedes_a180d',
                  'mercedes_b180d',
                  'mercedes_b200d',
                  'mercedes_b250e',
                  'mercedes_c180dt',
                  'mercedes_c180t',
                  'mercedes_c200d',
                  'mercedes_c200dt',
                  'mercedes_c220dt',
                  'mercedes_cl',
                  'mercedes_cla180',
                  'mercedes_cla220d',
                  'mercedes_cls',
                  'mercedes_e',
                  'mercedes_e200d',
                  'mercedes_e220d',
                  'mercedes_e280',
                  'mercedes_gla200d',
                  'mercedes_glc220d',
                  'mercedes_glc300',
                  'mercedes_sprinter']

TYPES_VW = ['vw_aventura',
            'vw_california',
            'vw_golf',
            'vw_id.3',
            'vw_multivan',
            'vw_passat',
            'vw_polo',
            'vw_scirocco',
            'vw_sharan',
            'vw_t-cross',
            'vw_t-roc',
            'vw_t6',
            'vw_tiguan',
            'vw_touran',
            'vw_transporter']

GRANULARITY = {'car': CAR,
               'manufacturer': MANUFACTURERS,
               'manufacturer_main': MANUFACTURERS_MAIN,
               'manufacturer_ford_vw': MANUFACTURERS_FORD_VW,
               'manufacturer_audi_mercedes': MANUFACTURERS_AUDI_MERCEDES,
               'type': TYPES,
               'type_audi': TYPES_AUDI,
               'type_volvo': TYPES_VOLVO,
               'type_ford': TYPES_FORD,
               'type_mercedes': TYPES_MERCEDES,
               'type_vw': TYPES_VW}


class LiDARCarDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, path, granularity, split=0.2, augmentation_split=False, strict_split=False, shuffle=True,
                 shuffle_pointcloud=False, normalize_distr=False, random_seed=None, color=False, augmented=True, *args,
                 **kwargs):
        try:
            self.labels = GRANULARITY[granularity]
        except KeyError:
            raise Exception('Invalid level of granularity: {}'.format(granularity))

        self.color = color

        super(LiDARCarDataset, self).__init__(*args, **kwargs)

        self.granularity = granularity
        self.split = split
        self.augmentation_split = augmentation_split
        self.strict_split = strict_split
        self.shuffle_pointcloud = shuffle_pointcloud
        self.files = []

        if isinstance(path, list):
            paths = path
        else:
            paths = [path]

        for path in paths:
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.ply') and self.is_in_labels(file):
                        filepath = os.path.join(subdir, file)
                        if augmented or 'augmented' not in filepath:
                            self.files.append(filepath)

        # check for duplicate filenames
        filenames = set()
        for filepath in self.files:
            filename = os.path.basename(filepath)
            if filename in filenames:
                raise Exception('Duplicate filename detected! The filenames of all examples must be unique.\n'
                                'The duplicate filename is: "' + filename + '"\n'
                                'One of the filepaths is: "' + filepath + '"')
            filenames.add(filename)

        # shuffle the dataset
        if shuffle:
            if random_seed is None:
                random_seed = random.randint(0, sys.maxsize)
            self.random_seed = random_seed
            random.seed(random_seed)
            random.shuffle(self.files)

        self.normalize_distr = normalize_distr
        if self.normalize_distr:
            distr, lowest_count = self.get_distr()
            for label, count in distr.items():
                for i in range(count - lowest_count):
                    self.remove_file_with_label(label)

        # create random seeds for point cloud shuffle
        self.shuffle_pointcloud_seeds = [random.randint(0, sys.maxsize) for _ in range(len(self.files))]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'pointcloud': tfds.features.Tensor(shape=(None, 6 if self.color else 3), dtype=tf.float32),
                'label': tfds.features.ClassLabel(names=self.labels),
            }),
            supervised_keys=('pointcloud', 'label')
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        test_files = []
        train_files = []
        if self.normalize_distr + self.augmentation_split + self.strict_split > 1:
            raise Exception('Invalid dataset builder arguments: too many split arguments are True')
        if self.normalize_distr:
            count = {label: 0 for label in self.labels}

            for file in self.files:
                label = self.extract_label(os.path.basename(file))
                if count[label] % self.split_to_every_xth_element(self.split) == 0:
                    test_files.append(file)
                else:
                    train_files.append(file)
                count[label] += 1
        elif self.augmentation_split:
            for file in self.files:
                if 'augmented' in file:
                    train_files.append(file)
                else:
                    test_files.append(file)
        elif self.strict_split:
            # construct test set
            for label in self.labels:
                temp = [f for f in self.files if 'augmented' not in f and self.is_label(f, label)]
                split = int(len(temp) * self.split)
                test_files += temp[:split]

            # construct train set
            not_in = False
            for f in self.files:
                # is it in
                for t in test_files:
                    if self.same_car(f, t):
                        not_in = True
                        break
                if not_in:
                    not_in = False
                    continue
                train_files.append(f)
        else:
            split = int(len(self.files) * self.split)
            test_files = self.files[:split]
            train_files = self.files[split:]

        return {
            'test': self._generate_examples(test_files),
            'train': self._generate_examples(train_files)
        }

    def _generate_examples(self, files):
        """Yields examples."""
        for i, filepath in enumerate(files):
            name = os.path.basename(filepath)
            yield name.encode('utf-8', 'replace').decode(), {
                'pointcloud': self.parse_pointcloud(filepath, self.shuffle_pointcloud_seeds[i]),
                'label': self.extract_label(name),
            }

    # make the dataset iterable
    def __iter__(self):
        return self._generate_examples(self.files)

    def parse_pointcloud(self, filepath, seed):
        points = []
        with open(filepath, 'r') as file:
            # header
            if file.readline().strip() != 'ply':
                raise Exception('Invalid ply header')
            line = file.readline().strip()
            while line != 'end_header':
                line = file.readline().strip()

            # data
            line = file.readline()
            while line:
                line = line.strip().split(' ')
                if not (6 <= len(line) <= 7):
                    raise Exception('Invalid line length')
                if self.color:
                    points.append(tuple(map(float, line[:6])))
                else:
                    points.append(tuple(map(float, line[:3])))
                line = file.readline()

        if self.shuffle_pointcloud:
            random.Random(seed).shuffle(points)

        return np.asmatrix(points, dtype=np.float32)

    def is_in_labels(self, name):
        name = name.lower()
        for label in self.labels:
            if label in name:
                return True

        return False

    def is_label(self, file, label):
        return label in os.path.basename(file.lower())

    def get_distr(self):
        distr = {label: 0 for label in self.labels}

        for file in self.files:
            distr[self.extract_label(os.path.basename(file))] += 1

        return distr, min(distr.values())

    def remove_file_with_label(self, label):
        for i, file in enumerate(self.files):
            if self.is_label(file, label):
                self.files.pop(i)
                return

    def split_to_every_xth_element(self, split):
        value = 0
        count = 0
        while value < 1:
            value += split
            count += 1

        return count

    def same_car(self, file1, file2):
        date1 = DATE_PATTERN.search(file1).group()
        date2 = DATE_PATTERN.search(file2).group()

        return date1 == date2

    def extract_label(self, name):
        if self.granularity == 'car':
            return 'yes'

        name = name.lower()
        for label in self.labels:
            if label in name:
                return label

        return DEFAULT_LABEL
