
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = Path("data/processed/processed.parquet")
DESTINATION = Path("data/processed/processed_sample.parquet")

SAMPLE_FRACTION = 0.10
BATCH_SIZE = 50_000
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    parquet = pq.ParquetFile(SOURCE)

    total_original = 0
    total_sample = 0

    writer = None

    try:
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
            total_original += batch.num_rows

            mask = rng.random(batch.num_rows) < SAMPLE_FRACTION
            selected = np.flatnonzero(mask)

            if len(selected) == 0:
                continue

            sample = batch.take(pa.array(selected))

            if writer is None:
                writer = pq.ParquetWriter(
                    DESTINATION,
                    sample.schema,
                    compression="snappy",
                )

            writer.write_batch(sample)
            total_sample += sample.num_rows

        print(f"Registros originales: {total_original:,}")
        print(f"Registros seleccionados: {total_sample:,}")
        print(f"Archivo generado: {DESTINATION}")

    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
