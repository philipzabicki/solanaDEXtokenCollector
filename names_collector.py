import requests
import solana
# from solana.rpc.async_api import AsyncClient
from solana.rpc.api import Client
from solders.pubkey import Pubkey
import solana.transaction
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from credentials import NODE_HTTP


# A_CLIENT = AsyncClient("https://solana-mainnet.core.chainstack.com/39a226275130ba6ff05c331a443e1f6a")
CLIENT = Client(NODE_HTTP)
SPL_TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
RAYDIUM_ADDRESS = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
GET_TXN_DELAY = 0.5
GET_SIGN_DELAY = 2
RERUN_DELAY = 600
LIMIT_SIZE = 1000


def getTxDetail(txSignature):
    time.sleep(GET_TXN_DELAY)
    return json.loads(CLIENT.get_transaction(solana.transaction.Signature.from_string(txSignature),
                                             max_supported_transaction_version=0).to_json())


def getAllTransactions(address, limit=None):
    last_signature = None
    part = 0
    collected_details = []

    while True:
        time.sleep(GET_SIGN_DELAY)
        txs = json.loads(
            CLIENT.get_signatures_for_address(address, before=last_signature, limit=min(1000, limit)).to_json())
        time.sleep(GET_SIGN_DELAY)
        signatures = [tx["signature"] for tx in txs["result"]]

        if not signatures:
            break

        for sig in signatures:
            collected_details.append(getTxDetail(sig))
            print(f'\r processing {sig}', end='')

        if (limit is not None) and (len(collected_details) >= limit):
            break
        last_signature = solana.transaction.Signature.from_string(signatures[-1])
        part += 1
    return collected_details


def collect_mint_addresses(txns):
    mint_addresses = []
    for txn in txns:
        pre = txn['result']['meta']['preTokenBalances']
        post = txn['result']['meta']['postTokenBalances']
        for i, j in zip(pre, post):
            mint_addresses.append(i['mint'])
            mint_addresses.append(j['mint'])
    return list(set(mint_addresses))


def get_token_info(pair_address):
    url = f"https://api.dexscreener.com/latest/dex/tokens/{pair_address}"
    response = requests.get(url).json()
    if response['pairs'] is None:
        return ('pepe', 'PEPE')  # Default values if the pair is not found
    else:
        name = response['pairs'][0]['baseToken']['name']
        symbol = response['pairs'][0]['baseToken']['symbol']
        return (name, symbol)


def get_tokens_info_parallel(pair_addresses, max_workers=8):
    tokens_info = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_address = {executor.submit(get_token_info, addr): addr for addr in pair_addresses}
        for future in as_completed(future_to_address):
            address = future_to_address[future]
            try:
                token_info = future.result()
                tokens_info.append(token_info)
            except Exception as exc:
                print(f"{address} generated an exception: {exc}")
                tokens_info.append(('Error', 'Error'))  # Error handling
    return tokens_info


def write_tokens_info_to_csv(tokens_info, file_path):
    with open(file_path, mode='a+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for name, symbol in tokens_info:
            writer.writerow([name, symbol])


if __name__ == '__main__':
    while True:
        try:
            txns = getAllTransactions(RAYDIUM_ADDRESS, limit=1000)
            print('')
            print('Collecting mint addresses...')
            mints = collect_mint_addresses(txns)
            print('Extracting token names and symbols...')
            tokens_info = get_tokens_info_parallel(mints,8)
            write_tokens_info_to_csv(tokens_info, 'data/names_and_symbols.csv')
            time.sleep(RERUN_DELAY)
        except:
            time.sleep(600)

