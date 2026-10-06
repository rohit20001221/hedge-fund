from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import json
import asyncio
import gzip

TOKEN = 256265
EXPIRY = "2026-10-13"

def get_range_strike_prices(atm: int):
    strikes = []
    for strike in range(atm - 50 * 10, atm + 50 * 10 + 1, 50):
        strikes.append(str(strike))

    return strikes

async def get_feed(ws):
    async with connect(
        "wss://wsrelay.sensibull.com/broker/1?consumerType=platform_pro", 
        origin="https://web.sensibull.com",
        user_agent_header="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
    ) as socket:
        await socket.recv()

        await socket.send(json.dumps(           {
            "msgCommand": "subscribe",
            "dataSource": "option-chain",
            "brokerId": 1,
            "tokens": [],
            "underlyingExpiry": [
                {
                "underlying": TOKEN,
                "expiry": EXPIRY
                }
            ],
            "uniqueId": ""
        }))

        while True:
            message = await socket.recv()

            packetType = message[0]
            packet = message[1:]
            if packetType == 3:
                data = gzip.decompress(packet[12:])
                data = json.loads(data)

                atm_strike = int(data["atm_strike"])
                strikes = get_range_strike_prices(atm_strike)

                chain = data["chain"]

                gex_data = dict()

                for strike in strikes:
                    option = chain[strike]

                    call_option = option["call"]
                    put_option = option["put"]

                    gamma = option["greeks"]["gamma"]

                    gex = gamma * (call_option["oi"] - put_option["oi"])
                    gex_data[strike] = gex

                await ws.send(json.dumps(gex_data))

async def main():
    server = await serve(get_feed, "localhost", 3206)

    print("[x] server running on ws://localhost:3206")
    await server.serve_forever()


asyncio.run(main())