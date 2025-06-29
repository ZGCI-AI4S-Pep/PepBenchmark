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

import gzip
import pickle
from itertools import product
from pathlib import Path

from loguru import logger
from pep.builder import MolBuilder
from pep.entity import ConnectionInfo, ParsedData, ParsedMonomerInfo
from pep.library import MonomerLibrary
from pep.parsers.biln_parser import BilnSerializer
from pep.parsers.helm_parser import HelmSerializer
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from tqdm import tqdm


def obj2bstr(obj):
    return gzip.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def bstr2obj(bstr):
    return pickle.loads(gzip.decompress(bstr))


NUM_NA = 20
NUM_NNA = 233
MIN_LEN = 2
MAX_LEN = 2


natural_lib = MonomerLibrary.from_sdf_file(
    "natural", "pep/resources/natural_monomers.sdf"
)
logger.warning(f"Natural library size: {len(natural_lib)}")
non_natural_lib = MonomerLibrary.from_sdf_file(
    "non-natural", "pep/resources/non-natural_monomers.sdf"
)
logger.warning(f"Non-natural library size: {len(non_natural_lib)}")
allowed_monomers = [i for i in natural_lib[:NUM_NA]] + [
    i for i in non_natural_lib[:NUM_NNA]
]
enumerate_lib = MonomerLibrary.from_monomer_list("enumerate", allowed_monomers)


biln_serializer = BilnSerializer(library=enumerate_lib)
helm_serializer = HelmSerializer(library=enumerate_lib)


def enumerate_parsed_data(num_aa):
    mols = []
    for idx, seq in tqdm(enumerate(product(allowed_monomers, repeat=num_aa), start=1)):
        # list monomer
        parsed_data = ParsedData()
        for idx, res in enumerate(seq):
            parsed_m = ParsedMonomerInfo(m_idx=idx, abbr=res.m_abbr, res=res)
            parsed_data.monomers.append(parsed_m)
        # add implicit connections
        for i in range(len(parsed_data.monomers) - 1):
            m1, m2 = parsed_data.monomers[i], parsed_data.monomers[i + 1]
            implicit_bond1 = ConnectionInfo(m_idx=m1.m_idx, rg_idx=1)
            implicit_bond2 = ConnectionInfo(m_idx=m2.m_idx, rg_idx=0)
            # Standard peptide bond: R2 (COOH) from m1 --- R1 (NH2) from m2
            parsed_data.implicit_bond_map[i].append(implicit_bond1)
            parsed_data.implicit_bond_map[i].append(implicit_bond2)
        # print(seq)
        mol = MolBuilder(parsed_data).build()
        biln_str = biln_serializer.serialize(parsed_data)
        helm_str = helm_serializer.serialize(parsed_data)
        # logger.critical(f"Serialized: {biln_str}, {helm_str}")
        mol.SetProp("biln", biln_str)
        mol.SetProp("helm", helm_str)
        mols.append(mol)
    return mols


mols = enumerate_parsed_data(2)
dst = Path(f"pep/resources/pep_enumeration_{NUM_NA}NA_{NUM_NNA}NNA")

# env = lmdb.open("pep/resources/pep_enumeration", map_size=1024**4*4, lock=False, readahead=False)
# txn = env.begin(write=True)

for idx, m in enumerate(tqdm(mols)):
    m_h = Chem.AddHs(m)
    assert m_h is not None, f"Failed to add Hs to {idx} molecule {m.GetProp('biln')}"
    id = rdDistGeom.EmbedMolecule(
        m_h,
        maxAttempts=0,
        randomSeed=-1,
        clearConfs=True,
        useRandomCoords=False,
        boxSizeMult=2,
        randNegEig=True,
        numZeroFail=1,
        coordMap={},
        forceTol=0.001,
        ignoreSmoothingFailures=False,
        enforceChirality=True,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        printExpTorsionAngles=False,
        useSmallRingTorsions=True,
        useMacrocycleTorsions=True,
        ETversion=2,
        useMacrocycle14config=True,
    )
    if id != 0:
        logger.warning(f"Failed to embed {idx} molecule {m_h.GetProp('biln')}")
        continue
    subdir = dst / f"{idx}"
    subdir.mkdir(parents=True, exist_ok=False)
    Chem.SDWriter(str(subdir / f"{idx}.sdf")).write(m_h)
    # ret = run(["xtb", f'{idx}.sdf', "--opt"], cwd=subdir, stdout=PIPE, stderr=PIPE, check=False)
    # if ret.returncode != 0:
    # logger.warning(f"Failed to optimize {idx} molecule {m_h.GetProp('biln')}")

    # with open(subdir / 'stdout.log', 'wb') as f:
    # f.write(ret.stdout)
    # with open(subdir / 'stderr.log', 'wb') as f:
    # f.write(ret.stderr)

    # txn.put(f"{idx}".encode(), obj2bstr(m_h))
# txn.commit()
# env.close()
