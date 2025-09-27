# api_server.py
import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

DB_FILE = "data/tokens_raw.db"       # ta sama ścieżka co w db_collector.py
app = FastAPI(title="Tokens API")

# — pozwalamy sobie łączyć się zdalnie z przeglądarki lub Postmana —
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # produkcyjnie wpisz konkretną domenę/IP
    allow_methods=["GET"],
    allow_headers=["*"],
)

def read_only_conn() -> str:
    """Open SQLite in read-only mode so writes z db_collector nie są blokowane."""
    return f"file:{DB_FILE}?mode=ro&cache=shared"

@app.get("/download-db")
async def download_db():
    async def file_chunks(path: str, chunk_size: int = 8192):
        with open(path, "rb") as f:
            while (block := f.read(chunk_size)):
                yield block
    headers = {"Content-Disposition": "attachment; filename=tokens_raw.db"}
    return StreamingResponse(
        file_chunks(DB_FILE), media_type="application/octet-stream", headers=headers
    )

@app.get("/dump")
async def dump_tokens():
    query = "SELECT * FROM tokens;"
    async with aiosqlite.connect(read_only_conn(), uri=True) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query) as cursor:

            # header
            first_row = await cursor.fetchone()
            if first_row is None:
                return StreamingResponse(iter(()), media_type="text/csv")
            cols = list(first_row.keys())
            yield_header = True

            async def iter_csv():
                import csv, io
                buff = io.StringIO()
                writer = csv.writer(buff)

                nonlocal yield_header
                if yield_header:
                    writer.writerow(cols)
                    yield buff.getvalue()
                    buff.seek(0); buff.truncate(0)
                    yield_header = False
                    writer.writerow(first_row)

                # stream remaining rows
                async for row in cursor:
                    writer.writerow(row)
                    yield buff.getvalue()
                    buff.seek(0); buff.truncate(0)

            headers = {"Content-Disposition": "attachment; filename=tokens.csv"}
            return StreamingResponse(iter_csv(), media_type="text/csv", headers=headers)
        
@app.get("/tokens-since")
async def tokens_since(created_after: str, limit: int = 100000, order: str = "ASC"):
    """
    Stream rows newer than created_after (format: 'YYYY-MM-DD HH:MM:SS' UTC) as CSV.
    order: ASC (oldest first, default) or DESC.
    """
    order_by = "ASC" if order.upper() not in ("ASC", "DESC") else order.upper()
    query = f"""
        SELECT *
        FROM tokens
        WHERE pairCreatedAt > ?
        ORDER BY pairCreatedAt {order_by}
        LIMIT ?
    """

    async with aiosqlite.connect(read_only_conn(), uri=True) as db:
        db.row_factory = aiosqlite.Row
        try:
            async with db.execute(query, (created_after, limit)) as cursor:
                first_row = await cursor.fetchone()
                if first_row is None:
                    return StreamingResponse(iter(()), media_type="text/csv")

                cols = list(first_row.keys())
                sent_header = False

                async def iter_csv():
                    import csv, io
                    buff = io.StringIO()
                    writer = csv.writer(buff)

                    nonlocal sent_header
                    if not sent_header:
                        writer.writerow(cols)
                        yield buff.getvalue()
                        buff.seek(0); buff.truncate(0)
                        sent_header = True
                        writer.writerow(first_row)

                    async for row in cursor:
                        writer.writerow(row)
                        yield buff.getvalue()
                        buff.seek(0); buff.truncate(0)

                headers = {"Content-Disposition": "attachment; filename=tokens_since.csv"}
                return StreamingResponse(iter_csv(), media_type="text/csv", headers=headers)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/tokens")
async def list_tokens(limit: int = 100, offset: int = 0):
    query = """
        SELECT *
        FROM tokens
        ORDER BY pairCreatedAt DESC
        LIMIT ? OFFSET ?;
    """
    async with aiosqlite.connect(read_only_conn(), uri=True) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(query, (limit, offset))
    return [dict(r) for r in rows]

@app.get("/token/{pair_address}")
async def get_token(pair_address: str):
    query = "SELECT * FROM tokens WHERE pairAddress = ?;"
    async with aiosqlite.connect(read_only_conn(), uri=True) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchone(query, (pair_address,))
    if row is None:
        raise HTTPException(status_code=404, detail="Token not found")
    return dict(row)

if __name__ == "__main__":
    # uruchom serwer na wszystkich interfejsach, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
