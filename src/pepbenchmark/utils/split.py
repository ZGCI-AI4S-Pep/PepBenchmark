# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


try:
    from hestia.partition import ccpart_random
    from hestia.similarity import sequence_similarity_mmseqs
except ImportError as err:
    raise ImportError("Please install hestia by 'pip install hestia-good'! ") from err


def random_split(df, frac, seed):
    """
    Randomly split a dataframe into three parts with a given fraction.

    Args:
        df (pd.DataFrame): dataframe
        frac (list): fraction of data to split, a list of train/valid/test fractions
        seed (int): random seed

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """

    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=seed)
    train_val = df.drop(test.index)
    train = train_val.sample(
        frac=train_frac / (train_frac + val_frac), replace=False, random_state=seed
    )
    valid = train_val.drop(train.index)

    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def homology_based_split(df, frac, sim_threshold, seed, seq_alighment="mmseq2"):
    train_frac, val_frac, test_frac = frac

    if seq_alighment == "mmseq2":
        sim_df = sequence_similarity_mmseqs(df, field_name="sequence")
        train, test, valid, partition_labs = ccpart_random(
            df,
            threshold=sim_threshold,
            test_size=test_frac,
            valid_size=val_frac,
            seed=seed,
            sim_df=sim_df,
        )
        return {
            "train": df.iloc[train, :].reset_index(drop=True),
            "valid": df.iloc[test, :].reset_index(drop=True),
            "test": df.iloc[valid, :].reset_index(drop=True),
        }


def cold_split(df, frac, sim_threshold, seed, entity="seq_protein"):
    pass
