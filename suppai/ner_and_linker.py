"""
Classes and methods for running NER and linking over downloaded articles

"""

from typing import List, Dict

import spacy
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.abbreviation import AbbreviationDetector


# only keep entities of the following types
KEEP_TYPES = {
    'T002',    # plants
    'T109',    # organic chemical
    'T116',    # amino Acid, peptide, or protein
    'T120',    # chemical viewed functionally
    'T121',    # pharmacologic substance
    'T125',    # hormone
    'T127',    # vitamin
    'T129',    # immunologic factor
    'T195',    # antibiotic
    'T196',    # element, ion, or isotopes
    'T197',    # inorganic chemical
    'T200'     # clinical drug
}

# only keep checking if top matches are less than this threshold
BETTER_SCORE_THRESHOLD = 0.95


# class for running scispacy NER and linking over abstracts and keeping matching linking results
class DrugSupplementLinker:
    def __init__(self):
        print('loading scispacy (takes a moment)...')

        # remove unused pipes from scispacy
        self.nlp = spacy.load("en_core_sci_sm", disable=["tagger", "parser", "textcat"])

        # add sentence tokenizer back in (need to add explicitly when removing dependency parser)
        self.nlp.add_pipe('sentencizer')

        # add abbreviation detection
        self.nlp.add_pipe('abbreviation_detector')

        # add UMLS linker
        # resolve_abbreviations uses abbreviation detection results for linking
        # filter_for_definitions allows linker to link to entities without definitions in UMLS
        # threshold determines which results to return (set to more stringent to improve precision)
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    def get_linked_entities(self, text: str, top_k=1) -> List[Dict]:
        """
        Link all entities to UMLS entities and return only relevant ones with identifiers and types
        :param text:
        :param top_k:
        :return:
        """
        doc = self.nlp(text)
        linker = self.nlp.get_pipe("scispacy_linker")

        # keep list of relevant entities (those with matching semantic types)
        entities_by_sentence = []

        for sent_id, sent in enumerate(doc.sents):

            relevant_ents = []

            for ent in sent.ents:

                # list of linked entities from scispacy output
                # linked_ents = [(linker.umls.cui_to_entity[cui], score) for cui, score in ent._.umls_ents[:top_k]]
                linked_ents = [(linker.kb.cui_to_entity[cui], score) for cui, score in ent._.kb_ents[:top_k]]

                # list of linked entities filtered by semantic type
                linked_cuis = []
                for ent_num, (linked_ent, score) in enumerate(linked_ents):
                    if KEEP_TYPES:
                        # if the type matches, keep
                        if linked_ent.types and set(linked_ent.types).intersection(KEEP_TYPES):
                            linked_cuis.append([linked_ent.concept_id, linked_ent.types, score, ent_num])
                        # if the type doesn't match and the score is very high, skip this span
                        # this eliminates general text spans that match better to a different entity
                        elif linked_ent.types:
                            if score > BETTER_SCORE_THRESHOLD:
                                break
                    else: # keep everything if no filter specified
                        linked_cuis.append([linked_ent.concept_id, linked_ent.types, score, ent_num])

                # keep entity if at least some were same semantic type
                if linked_cuis:
                    relevant_ents.append({
                        "string": ent.text,
                        "start": ent.start_char - sent.start_char,
                        "end": ent.end_char - sent.start_char,
                        "linked_cuis": linked_cuis
                    })

            if relevant_ents:
                entities_by_sentence.append({
                    "sent_num": sent_id,
                    "sentence": sent.text,
                    "entities": relevant_ents
                })

        return entities_by_sentence