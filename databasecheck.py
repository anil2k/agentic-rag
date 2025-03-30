import asyncio
import asyncpg

DB_URL = "postgresql://ai:ai@localhost:5532/ai"
#db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


async def test_connection():
    try:
        conn = await asyncpg.connect(DB_URL)
        version = await conn.fetchval("SELECT version();")
        print("✅ Connected to PostgreSQL:", version)
        await conn.close()
    except Exception as e:
        print("❌ Connection failed:", e)

asyncio.run(test_connection())
