import aiohttp
import asyncio

TELEGRAM_TOKEN = '7726253154:AAEOXD9jJzaU2j5Gk-hwWbSonr7lQQhKHrk'

async def get_updates():
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main():
    updates = await get_updates()
    print(updates)

if __name__ == "__main__":
    asyncio.run(main())
