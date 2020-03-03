import neurosynth
from neurosynth import Dataset


neurosynth.dataset.download(path='data/external/neurosynth/.', unpack=True)

dataset = Dataset('data/external/neurosynth/database.txt')

dataset.save('data/external/neurosynth/dataset.pkl')  # load this in the future
