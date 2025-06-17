
from .convert import Peptide
from .featurizer import PeptideFeaturizer
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

tqdm.pandas(desc="Converting sequences")

def PepConvert(df, original_format, convert_format, feature_type, all_nature):
    #
    if convert_format is not None:

        df[convert_format] = df['sequence'].apply(
            lambda s: Peptide(s, format=original_format, all_nature=all_nature).to(convert_format)
        )

    # 特征提取
    if feature_type is not None:
        featurizer = PeptideFeaturizer(
            input_format=original_format,
            feature_type=feature_type,
            is_natural=all_nature
        )
        features = featurizer(df['sequence'].tolist())


        if isinstance(features, list):
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(features)], axis=1)
        elif isinstance(features, np.ndarray):
            for i in range(features.shape[1]):
                df[f'feat_{i}'] = features[:, i]
        elif isinstance(features, pd.DataFrame):
            df = pd.concat([df.reset_index(drop=True), features], axis=1)
    return df