import json
import unittest


INTERACTION_DICT_FILE = 'output/interaction_id_dict.json'
SENTENCE_DICT_FILE = 'output/sentence_dict.json'
CUI_METADATA_FILE = 'output/cui_metadata.json'
PAPER_METADATA_FILE = 'output/paper_metadata.json'


class TestDataIntegrity(unittest.TestCase):

    def setUp(self):
        with open(INTERACTION_DICT_FILE, 'r') as f:
            self.interaction_dict = json.load(f)
        with open(SENTENCE_DICT_FILE, 'r') as f:
            self.sentence_dict = json.load(f)
        with open(CUI_METADATA_FILE, 'r') as f:
            self.cui_metadata = json.load(f)
        with open(PAPER_METADATA_FILE, 'r') as f:
            self.paper_metadata = json.load(f)

    def test_cui_keys(self):
        """
        Assert all cui keys in interaction dict have metadata
        :return:
        """
        for cui, interactions in self.interaction_dict.items():
            assert cui in self.cui_metadata

    def test_cui_keys_reverse(self):
        """
        Assert all cui keys in metadata have interaction dict entries
        :return:
        """
        for cui, metadata in self.cui_metadata.items():
            assert cui in self.interaction_dict

    def test_interaction_keys(self):
        """
        Assert all interaction keys are in sentence dict
        :return:
        """
        for cui, interactions in self.interaction_dict.items():
            for interaction in interactions:
                assert interaction in self.sentence_dict

    def test_papers_in_dict(self):
        """
        Assert all papers in sentence dict have entries in paper_metadata
        :return:
        """
        for sentences in self.sentence_dict.values():
            for sent in sentences:
                assert sent["paper_id"] in self.paper_metadata

    def test_no_redundant_names(self):
        """
        Assert all preferred names are unique
        :return:
        """
        pref_name = [meta['preferred_name'] for meta in self.cui_metadata.values()]
        assert len(pref_name) == len(set(pref_name))

    def test_all_cuis_in_sentences_valid(self):
        """
        Assert all CUI pairs in sentences are valid
        :return:
        """
        for int_key, sentences in self.sentence_dict.items():
            cui1, cui2 = int_key.split('-')
            for sent in sentences:
                arg1 = sent['arg1']['id']
                arg2 = sent['arg2']['id']
                assert {cui1, cui2} == {arg1, arg2}
                assert arg1 in self.cui_metadata
                assert arg2 in self.cui_metadata

    def test_no_empty_entries(self):
        """
        Assert all entries contain data
        :return:
        """
        for sentences in self.sentence_dict.values():
            assert sentences
        for metadata in self.cui_metadata.values():
            assert metadata
        for paper in self.paper_metadata.values():
            assert paper