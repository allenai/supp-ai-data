from collections import defaultdict
from typing import List, Dict

from s2base2.config import DB_S2_CORPUS
from s2base2.db_utils import S2DBIterator

from suppai.data import PaperAuthor


def get_paper_metadata(pids: List[str], query_authors=False) -> Dict[str, Dict]:
    """
    Get metadata entries from papers table
    default: do not query authors
    """

    metadata_query = """
        SELECT l.sha, p.id, p.title, p.year, p.venue, p.doi, p.pmid, p.fields_of_study
        FROM papers p
        INNER JOIN legacy_paper_ids l ON l.paper_id = p.id
        WHERE p.id in ({}) AND l.id_type = 'Canonical';
    """.format(','.join([f"'{pid}'" for pid in pids]))

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

    # add author info if requested
    if query_authors:
        new_paper_ids = list(s2_id_to_hash.keys())
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


def get_paper_metadata_no_sha(pids: List[str]) -> Dict[str, Dict]:
    """
    Get metadata entries from papers table without SHA
    """

    metadata_query = """
        SELECT p.id, p.title, p.year, p.venue, p.doi, p.pmid, p.fields_of_study
        FROM papers p
        WHERE p.id in ({});
    """.format(','.join([f"'{pid}'" for pid in pids]))

    s2_id_to_metadata = dict()
    for pid, title, year, venue, doi, pmid, fos in S2DBIterator(query_text=metadata_query, db_config=DB_S2_CORPUS):
        s2_id_to_metadata[str(pid)] = {
            "title": title,
            "authors": [],
            "year": year,
            "venue": venue,
            "doi": doi,
            "pmid": pmid,
            "fields_of_study": fos
        }
    return s2_id_to_metadata