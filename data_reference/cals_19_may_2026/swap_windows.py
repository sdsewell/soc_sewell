"""
swap_windows.py — Windows replacement for swap.sh

Replicates:  objcopy -I binary -O binary --reverse-bytes=8 <in> <out>

For every 8-byte chunk in the input file the byte order is reversed:
  [b0, b1, b2, b3, b4, b5, b6, b7]  →  [b7, b6, b5, b4, b3, b2, b1, b0]

Processes all *.bin files in the same folder that do NOT already end in
_swapped.bin, writing <stem>_swapped.bin alongside the originals.
"""

import pathlib
import numpy as np

CHUNK = 8
FOLDER = pathlib.Path(__file__).parent


def reverse_bytes_8(src: pathlib.Path, dst: pathlib.Path) -> None:
    data = np.frombuffer(src.read_bytes(), dtype=np.uint8)
    n = len(data)
    pad = (CHUNK - n % CHUNK) % CHUNK
    if pad:
        data = np.pad(data, (0, pad))
    swapped = data.reshape(-1, CHUNK)[:, ::-1].flatten()
    if pad:
        swapped = swapped[:-pad]
    dst.write_bytes(swapped.tobytes())


def main() -> None:
    bins = sorted(
        f for f in FOLDER.glob("*.bin") if not f.stem.endswith("_swapped")
    )
    if not bins:
        print("No .bin files found to process.")
        return

    for src in bins:
        dst = src.with_name(src.stem + "_swapped.bin")
        print(f"  {src.name}  →  {dst.name}", end="  ", flush=True)
        reverse_bytes_8(src, dst)
        print(f"({dst.stat().st_size} bytes)")

    print(f"\nDone. {len(bins)} file(s) processed.")


if __name__ == "__main__":
    main()
