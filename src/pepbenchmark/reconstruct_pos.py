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

import pandas as pd

from pepbenchmark.raw_data import DATASET_MAP

dataset_names = list(DATASET_MAP.keys())
dataset_names = ["hemolytic"]

for dataset in dataset_names:
    file_path = DATASET_MAP.get(dataset, {}).get("path", None)
    df = pd.read_csv(file_path + "/pos_seqence.csv")
    sequence_origin = df["sequence"].tolist()
    property_name = DATASET_MAP.get(dataset, {}).get("property_name", None)
    if property_name:
        print(f"Processing dataset {dataset} with property {property_name}.")
        if isinstance(property_name, str):
            property_name = [property_name]
        included_activities = [a.lower() for a in property_name]
        pos_pool_path = "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/negative_pool/peptidepedia.csv"
        pos_pool = pd.read_csv(pos_pool_path)
        pos_pool.columns = [col.lower() for col in pos_pool.columns]
        mask = pos_pool[included_activities].sum(axis=1) >= 1
        sequence_add = pos_pool[mask]["sequence"].tolist()
        sequence = sequence_origin + sequence_add
    else:
        sequence = sequence_origin
    # 只保留长度小于等于50的序列
    # sequence = [s for s in sequence if len(s) <= 50]
    seq_deduplicated = list(set(sequence))
    df_deduplicated = pd.DataFrame({"sequence": seq_deduplicated})
    df_deduplicated.to_csv(file_path + "/pos_seqs.csv", index=False)
    print(
        f"Finish processing dataset {dataset}.origin_seq {len(sequence_origin)}, add_seq {len(sequence_add)}, get {len(seq_deduplicated)} positive sequences."
    )
