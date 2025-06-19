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


def sample_from_bioactive():
    """"""


def neg_sample(df, frac, column_names, two_types=False):
    """
    x = int(len(df) * frac)
    id1, x1, id2, x2 = column_names
    df[id1] = df[id1].apply(lambda x: str(x))
    df[id2] = df[id2].apply(lambda x: str(x))

    if not two_types:
        df_unique = np.unique(df[[id1, id2]].values.reshape(-1))
        pos = df[[id1, id2]].values
        pos_set = set([tuple([i[0], i[1]]) for i in pos])
        np.random.seed(1234)
        samples = np.random.choice(df_unique, size=(x, 2), replace=True)
        neg_set = set([tuple([i[0], i[1]]) for i in samples if i[0] != i[1]
                      ]) - pos_set
        neg_set = set([tuple([i[0], i[1]]) for i in samples if i[0] != i[1]
                      ]) - pos_set

        while len(neg_set) < x:
            sample = np.random.choice(df_unique, 2, replace=False)
            sample = tuple([sample[0], sample[1]])
            if sample not in pos_set:
                neg_set.add(sample)
        neg_list = [list(i) for i in neg_set]

        id2seq = dict(df[[id1, x1]].values)
        id2seq.update(df[[id2, x2]].values)

        neg_list_val = []
        for i in neg_list:
            neg_list_val.append([i[0], id2seq[i[0]], i[1], id2seq[i[1]], 0])

        concat_frames = pd.DataFrame(neg_list_val).rename(columns={
            0: id1,
            1: x1,
            2: id2,
            3: x2,
            4: "Y"
        })
        df = pd.concat([df, concat_frames], ignore_index=True, sort=False)
        return df
    else:
        df_unique_id1 = np.unique(df[id1].values.reshape(-1))
        df_unique_id2 = np.unique(df[id2].values.reshape(-1))

        pos = df[[id1, id2]].values
        pos_set = set([tuple([i[0], i[1]]) for i in pos])
        np.random.seed(1234)

        sample_id1 = np.random.choice(df_unique_id1, size=len(df), replace=True)
        sample_id2 = np.random.choice(df_unique_id2, size=len(df), replace=True)

        neg_set = (set([
            tuple([sample_id1[i], sample_id2[i]])
            for i in range(len(df))
            if sample_id1[i] != sample_id2[i]
        ]) - pos_set)
        neg_set = (set([
            tuple([sample_id1[i], sample_id2[i]])
            for i in range(len(df))
            if sample_id1[i] != sample_id2[i]
        ]) - pos_set)

        while len(neg_set) < len(df):
            sample_id1 = np.random.choice(df_unique_id1, size=1, replace=True)
            sample_id2 = np.random.choice(df_unique_id2, size=1, replace=True)

            sample = tuple([sample_id1[0], sample_id2[0]])
            if sample not in pos_set:
                neg_set.add(sample)
        neg_list = [list(i) for i in neg_set]

        id2seq1 = dict(df[[id1, x1]].values)
        id2seq2 = dict(df[[id2, x2]].values)

        neg_list_val = []
        for i in neg_list:
            neg_list_val.append([i[0], id2seq1[i[0]], i[1], id2seq2[i[1]], 0])

        df = pd.concat([
            df,
            pd.DataFrame(neg_list_val).rename(columns={
                0: id1,
                1: x1,
                2: id2,
                3: x2,
                4: "Y"
            })
        ],
                       ignore_index=True,
                       sort=False)
        return df"""
