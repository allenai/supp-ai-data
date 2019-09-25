import json
import unittest

from s2base2.list_utils import flatten


CUI_FILE = 'data/cui_clusters.json'


class TestCUIIntegrity(unittest.TestCase):

    def setUp(self):
        with open(CUI_FILE, 'r') as f:
            self.cui_dict = json.load(f)

    def test_supp_clusters_include_keys(self):
        """
        Assert supp clusters include cluster keys
        :return:
        """
        for k, v in self.cui_dict["supplements"].items():
            assert k in v['members']

    def test_drug_clusters_include_keys(self):
        """
        Assert drug clusters include cluster keys
        :return:
        """
        for k, v in self.cui_dict["drugs"].items():
            assert k in v['members']

    def test_no_supp_drug_overlaps(self):
        """
        Assert supplement and drug identifiers don't overlap
        :return:
        """
        supp_cuis = flatten([v['members'] for v in self.cui_dict["supplements"].values()])
        drug_cuis = flatten([v['members'] for v in self.cui_dict["drugs"].values()])
        assert not set(supp_cuis) & set(drug_cuis)

    def test_no_supp_cluster_overlaps(self):
        """
        Assert no supp clusters have overlaps
        :return:
        """
        done_cuis = set([])
        for v in self.cui_dict["supplements"].values():
            assert not set(v['members']) & done_cuis
            done_cuis.update(set(v['members']))

    def test_no_drug_cluster_overlaps(self):
        """
        Assert no drug clusters have overlaps
        :return:
        """
        done_cuis = set([])
        for v in self.cui_dict["drugs"].values():
            assert not set(v['members']) & done_cuis
            done_cuis.update(set(v['members']))
