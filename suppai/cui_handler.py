"""
Handles supplement and drug CUIs and merging logic

"""

import os
import json
from suppai.data import CUIMetadata


CUI_FILE = 'data/cui_clusters.json'


class CUIHandler:
    def __init__(
            self,
            cluster_file=CUI_FILE
    ):
        assert os.path.exists(cluster_file)

        with open(cluster_file, 'r') as f:
            self.cluster_dict = json.load(f)

        self._build_dicts()

    def _build_dicts(self):
        """
        Build dictionaries for mapping entity CUIs
        :return:
        """
        self.supps = set([])
        self.drugs = set([])
        self.map_dict = dict()

        for cluster_key, cluster_members in self.cluster_dict['supplements'].items():
            self.supps.add(cluster_key)
            for mem in cluster_members['members']:
                if mem in self.map_dict and self.map_dict[mem] != cluster_key:
                    raise Exception("Overlapping clusters!")
                self.map_dict[mem] = cluster_key

        for cluster_key, cluster_members in self.cluster_dict['drugs'].items():
            self.drugs.add(cluster_key)
            for mem in cluster_members['members']:
                if mem in self.map_dict and self.map_dict[mem] != cluster_key:
                    raise Exception("Overlapping clusters!")
                self.map_dict[mem] = cluster_key

        self.valid_cuis = set(self.map_dict.keys())

    def form_cui_entry(self, cui):
        """
        Form CUI metadata entry
        :param cui:
        :return:
        """
        if cui in self.supps:
            return CUIMetadata(
                ent_type="supplement",
                preferred_name=self.cluster_dict['supplements'][cui]['preferred_name'],
                synonyms=self.cluster_dict['supplements'][cui]['synonyms'],
                tradenames=[],
                definition=self.cluster_dict['supplements'][cui]['definition']
            )
        elif cui in self.drugs:
            return CUIMetadata(
                ent_type="drugs",
                preferred_name=self.cluster_dict['drugs'][cui]['preferred_name'],
                synonyms=self.cluster_dict['drugs'][cui]['synonyms'],
                tradenames=self.cluster_dict['drugs'][cui]['tradenames'],
                definition=self.cluster_dict['drugs'][cui]['definition']
            )
        else:
            raise KeyError("Invalid supplement or drug CUI!")

    def is_valid_cui(self, cui: str):
        return cui in self.valid_cuis

    def get_cui_type(self, cui: str) -> str:
        if cui not in self.map_dict:
            return ""
        if self.map_dict[cui] in self.supps:
            return "supplement"
        if self.map_dict[cui] in self.drugs:
            return "drug"
        return ""

    def normalize_cui(self, cui: str) -> str:
        if cui not in self.map_dict:
            return ""
        return self.map_dict[cui]

    def is_supp_drug(self, cui1: str, cui2: str) -> bool:
        if cui1 not in self.map_dict:
            return False
        if cui2 not in self.map_dict:
            return False
        if {self.get_cui_type(cui1), self.get_cui_type(cui2)} == {"supplement", "drug"}:
            return True
        else:
            return False

    def is_supp_supp(self, cui1: str, cui2: str) -> bool:
        if cui1 not in self.map_dict:
            return False
        if cui2 not in self.map_dict:
            return False
        if self.get_cui_type(cui1) == "supplement" and self.get_cui_type(cui2) == "supplement":
            return True
        else:
            return False