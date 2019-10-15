from collections import defaultdict
from typing import List, Dict

from s2base2.config import DB_S2_CORPUS
from s2base2.db_utils import S2DBIterator

from suppai.data import PaperAuthor


def get_paper_metadata(s2_ids: List[str]) -> Dict[str, Dict]:
    """Get metadata entries from papers table"""

    metadata_query = """
        SELECT t1.sha, t2.id, t2.title, t2.year, t2.venue, t2.doi, t2.pmid, t2.fields_of_study, t3.source
        FROM legacy_paper_ids t1 
        INNER JOIN papers t2 ON t1.paper_id = t2.id 
        INNER JOIN sourced_paper t3 ON t1.paper_id = t3.id
        WHERE t1.sha in ({});
    """.format(','.join([f"'{s2_id}'" for s2_id in s2_ids]))

    s2_id_to_metadata = dict()
    s2_id_to_hash = dict()
    for row in S2DBIterator(query_text=metadata_query, db_config=DB_S2_CORPUS):
        s2_id_to_hash[row[1]] = row[0]
        s2_id_to_metadata[row[0]] = {
            "title": row[2],
            "authors": [],
            "year": row[3],
            "venue": row[4],
            "doi": row[5],
            "pmid": row[6],
            "fields_of_study": row[7]
        }

    new_paper_ids = list(s2_id_to_hash.keys())

    # add author info
    author_query = """
        SELECT t1.paper_id, t1.position, t1.author_id, t2.first_name, t2.middle_names, t2.last_name, t2.suffix 
        FROM paper_authors t1 INNER JOIN author t2 ON t1.author_id = t2.id 
        WHERE paper_id IN ({});
    """.format(','.join([f"'{new_id}'" for new_id in new_paper_ids]))

    authors = defaultdict(list)
    for row in S2DBIterator(query_text=author_query, db_config=DB_S2_CORPUS):
        authors[row[0]].append([row[1], row[3], row[4], row[5], row[6]])

    for new_id, author_entries in authors.items():
        author_entries.sort(key=lambda x: x[0])
        author_list = []
        for entry in author_entries:
            author_list.append(PaperAuthor(
                first=entry[1],
                middle=' '.join(entry[2]) if entry[2] else None,
                last=entry[3],
                suffix=entry[4]
            ))
        s2_id_to_metadata[s2_id_to_hash[new_id]]["authors"] = author_list

    return s2_id_to_metadata