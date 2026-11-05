"""Ejemplo real: python demo_async.py [--stress]."""
import argparse
import asyncio
import logging

from chain import process_text


async def main(stress: bool = False) -> int:
    texts = ["La API FastAPI usa Redis y PostgreSQL. El servicio está caído por "
             "agotamiento de conexiones concurrentes a PostgreSQL."]
    if stress:
        texts += ["Algo anda raro, quizás sea la red; no hay logs ni tecnologías identificadas."]
    failed = False
    for index, text in enumerate(texts, start=1):
        print(f"Caso {index}")
        try:
            result = await process_text(text)
            print(result.model_dump_json(indent=2))
        except Exception as exc:
            failed = True
            print(f"No se obtuvo un objeto validado ({type(exc).__name__}). Revisá configuración y logs.")
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        raise SystemExit(asyncio.run(main(args.stress)))
    except KeyboardInterrupt:
        raise SystemExit(130)
