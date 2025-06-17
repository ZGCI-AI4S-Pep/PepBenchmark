from src.pepbenchmark.dataset_loader import SingleTaskDataset
import warnings
warnings.filterwarnings("ignore")


data = SingleTaskDataset('QS_APML',
                     convert_format = 'smiles',
                     feature_type = 'ecfp',
                     )

data.get_metadata()
data.pep_distribution()
train = data.get_split(method= 'random')['train']
print(train.shape)
data.print_stats()
data.pep_distribution()







