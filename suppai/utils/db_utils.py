from collections import defaultdict
from typing import List, Dict

from suppai.data import PaperAuthor


def get_paper_metadata_no_sha(pids: List[str]) -> Dict[str, Dict]:
    """
    Get metadata entries from papers table without SHA
    """

    raise NotImplementedError("This function is not implemented yet.")
    # reimplement with API
    # s2_id_to_metadata = dict()
    # for pid, title, year, venue, doi, pmid, fos in db_entries:
    #     s2_id_to_metadata[str(pid)] = {
    #         "title": title,
    #         "authors": [],
    #         "year": year,
    #         "venue": venue,
    #         "doi": doi,
    #         "pmid": pmid,
    #         "fields_of_study": fos
    #     }
    # return s2_id_to_metadata