"""Retrieval over stored chunks: embed the query, rank by cosine similarity.

Brute-force cosine in numpy is plenty fast for a corpus of a few thousand
chunks (a handful of cities' worth of council meetings) - no vector DB
needed yet. Revisit if the corpus grows into the tens of thousands.
"""
import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from app.rag.embed import embed_text


@dataclass
class RetrievedChunk:
    citation_id: str
    city: str
    meeting_date: str | None
    meeting_title: str
    start_ts: float
    end_ts: float
    text: str
    score: float


def search_documents(
    db: sqlite3.Connection, query: str, top_k: int = 8, city: str | None = None
) -> list[RetrievedChunk]:
    query_vec = embed_text(query)

    sql = "SELECT id, city, meeting_date, meeting_title, start_ts, end_ts, text, embedding FROM chunks"
    params: tuple = ()
    if city:
        sql += " WHERE city = ?"
        params = (city,)

    rows = db.execute(sql, params).fetchall()
    if not rows:
        return []

    scored = []
    for row in rows:
        vec = np.array(json.loads(row[7]))
        score = float(
            np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        )
        scored.append((score, row))

    scored.sort(key=lambda pair: -pair[0])

    return [
        RetrievedChunk(
            citation_id=f"C{row[0]}",
            city=row[1],
            meeting_date=row[2],
            meeting_title=row[3],
            start_ts=row[4],
            end_ts=row[5],
            text=row[6],
            score=score,
        )
        for score, row in scored[:top_k]
    ]
