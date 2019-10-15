from typing import List, NamedTuple, Optional


class CUIMetadata(NamedTuple):
    """
    Class for storing CUI metadata from UMLS
    """
    ent_type: str
    preferred_name: str
    synonyms: List[str]
    tradenames: List[str]
    definition: str

    def as_json(self):
        return {
            "ent_type": self.ent_type,
            "preferred_name": self.preferred_name,
            "synonyms": self.synonyms,
            "tradenames": self.tradenames,
            "definition": self.definition
        }


class PaperAuthor(NamedTuple):
    """
    Class for storing Paper authors
    """
    first: Optional[str]
    middle: Optional[str]
    last: str
    suffix: Optional[str]

    def as_json(self):
        return {
            "first": self.first,
            "middle": self.middle,
            "last": self.last,
            "suffix": self.suffix
        }


class PaperMetadata(NamedTuple):
    """
    Class for storing Paper metadata from S2
    """
    title: str
    authors: List[PaperAuthor]
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    pmid: int
    fields_of_study: List[str]
    retraction: bool
    clinical_study: bool
    human_study: bool
    animal_study: bool

    def as_json(self):
        return {
            "title": self.title,
            "authors": [author.as_json() for author in self.authors],
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "pmid": self.pmid,
            "fields_of_study": self.fields_of_study,
            "retraction": self.retraction,
            "clinical_study": self.clinical_study,
            "human_study": self.human_study,
            "animal_study": self.animal_study
        }


class LabeledSpan(NamedTuple):
    """
    Class for labeled spans
    """
    id: str
    span: List[int]

    def as_json(self):
        return {
            "id": self.id,
            "span": self.span
        }


class EvidenceSentence(NamedTuple):
    """
    Class for storing evidence sentence
    """
    uid: int
    paper_id: str
    sentence_id: int
    sentence: str
    confidence: Optional[float]
    arg1: LabeledSpan
    arg2: LabeledSpan

    def as_json(self):
        return {
            "uid": self.uid,
            "paper_id": self.paper_id,
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "confidence": self.confidence,
            "arg1": self.arg1.as_json(),
            "arg2": self.arg2.as_json()
        }