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

from abc import ABC

from pep.entity import ParsedData


class Parser(ABC):
    """
    Base class for all parsers.
    """

    def __init__(self, monomer_library):
        self.library = monomer_library

    def parse(self, *args, **kwargs) -> ParsedData:
        """
        Parse the input data.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Serializer(ABC):
    """
    Base class for all serializers.
    """

    def __init__(self, monomer_library):
        self.library = monomer_library

    def serialize(self, *args, **kwargs) -> str:
        """
        Serialize the parsed data to a string.
        """
        raise NotImplementedError("Subclasses must implement this method.")
