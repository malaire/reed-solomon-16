# reed-solomon-16

A library for Reed-Solomon `GF(2^16)` erasure coding, featuring:

- `O(n log n)` complexity.
- Any combination of 1 - 32768 original shards with 1 - 32768 recovery shards.
- Up to 65535 original or recovery shards with some limitations.
- SIMD optimizations are planned, but not yet implemented.

## Simple usage

1. Divide data into equal-sized original shards.
   Shard size must be multiple of 64 bytes.
2. Decide how many recovery shards you want.
3. Generate recovery shards with [`reed_solomon_16::encode`].
4. When some original shards get lost, restore them with [`reed_solomon_16::decode`].
    - You must provide at least as many shards as there were original shards in total,
      in any combination of original shards and recovery shards.

### Example

Divide data into 3 original shards of 64 bytes each and generate 5 recovery shards.
Assume then that original shards #0 and #2 are lost
and restore them by providing 1 original shard and 2 recovery shards.

```rust
let original = [
    b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
    b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
    b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
];

let recovery = reed_solomon_16::encode(
    3, // total number of original shards
    5, // total number of recovery shards
    original, // all original shards
)?;

let restored = reed_solomon_16::decode(
    3, // total number of original shards
    5, // total number of recovery shards
    [  // provided original shards with indexes
        (1, &original[1]),
    ],
    [  // provided recovery shards with indexes
        (1, &recovery[1]),
        (4, &recovery[4]),
    ],
)?;

assert_eq!(restored[&0], original[0]);
assert_eq!(restored[&2], original[2]);
# Ok::<(), reed_solomon_16::Error>(())
```

## Basic usage

[`ReedSolomonEncoder`] and [`ReedSolomonDecoder`] give more control
of the encoding/decoding process.

Here's the above example using these instead:

```rust
use reed_solomon_16::{ReedSolomonDecoder, ReedSolomonEncoder};
use std::collections::HashMap;

let original = [
    b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
    b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
    b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
];

let mut encoder = ReedSolomonEncoder::new(
    3, // total number of original shards
    5, // total number of recovery shards
    64, // shard size in bytes
)?;

for original in original {
    encoder.add_original_shard(original)?;
}

let result = encoder.encode()?;
let recovery: Vec<_> = result.recovery_iter().collect();

let mut decoder = ReedSolomonDecoder::new(
    3, // total number of original shards
    5, // total number of recovery shards
    64, // shard size in bytes
)?;

decoder.add_original_shard(1, original[1])?;
decoder.add_recovery_shard(1, recovery[1])?;
decoder.add_recovery_shard(4, recovery[4])?;

let result = decoder.decode()?;
let restored: HashMap<_, _> = result.restored_original_iter().collect();

assert_eq!(restored[&0], original[0]);
assert_eq!(restored[&2], original[2]);
# Ok::<(), reed_solomon_16::Error>(())
```

## Advanced usage

See [`rate`][mod:rate] module for advanced encoding/decoding
using chosen [`Engine`] and [`Rate`].

## Benchmarks

- These benchmarks are from `cargo bench main`
  with 3.4 GHz i5-3570K (Ivy Bridge, 3rd gen.).
- Shards are 1024 bytes.
- MiB/s is total amount of data,
  i.e. original shards + recovery shards.
    - For decoder this includes missing shards.
- Encode benchmark
    - Includes [`add_original_shard`][RSE::add_original_shard] and
      [`encode`][RSE::encode] of [`ReedSolomonEncoder`].
- Decode benchmark
    - Has two MiB/s values for 1% and 100% original shard loss, of maximum possible.
    - Provides minimum required amount of shards to decoder.
    - Includes [`add_original_shard`][RSD::add_original_shard],
      [`add_recovery_shard`][RSD::add_recovery_shard] and
      [`decode`][RSD::decode] of [`ReedSolomonDecoder`].

| original : recovery | MiB/s (encode) | MiB/s (decode) |
| ------------------- | -------------- | -------------- |
| 100 : 100           | 229            | 73 ; 71        |
| 100 : 1 000         | 229            | 66 ; 66        |
| 1 000 : 100         | 222            | 65 ; 64        |
| 1 000 : 1 000       | 171            | 77 ; 74        |
| 1 000 : 10 000      | 149            | 53 ; 53        |
| 10 000 : 1 000      | 154            | 55 ; 55        |
| 10 000 : 10 000     | 103            | 39 ; 38        |
| 16 385 : 16 385     |  89            | 31 ; 31        |
| 32 768 : 32 768     | 107            | 50 ; 49        |

## Benchmarks against other crates

Use `cargo run --release --example quick-comparison`
to run few simple benchmarks against [`reed-solomon-erasure`]
and [`reed-solomon-novelpoly`] crates.

This crate is fastest when shard count exceeds 256 shards,
except for one-time initialization (< 10 ms)
which can dominate at really small data amounts.

[`reed-solomon-erasure`]: https://crates.io/crates/reed-solomon-erasure
[`reed-solomon-novelpoly`]: https://crates.io/crates/reed-solomon-novelpoly

## Running tests

Some larger tests are marked `#[ignore]` and are not run with `cargo test`.
Use `cargo test -- --ignored` to run those.

## Safety

This crate doesn't currently use any `unsafe` code.

However planned SIMD-optimized engines will need to use `unsafe`,
but the intention is that nothing else will use `unsafe`.

## Credits

This crate is based on [Leopard-RS] by Christopher A. Taylor.

[Leopard-RS]: https://github.com/catid/leopard

[`Naive`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/engine/struct.Naive.html
[`NoSimd`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/engine/struct.NoSimd.html

[`ReedSolomonEncoder`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/struct.ReedSolomonEncoder.html
[RSE::add_original_shard]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/struct.ReedSolomonEncoder.html#method.add_original_shard
[RSE::encode]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/struct.ReedSolomonEncoder.html#method.encode

[`ReedSolomonDecoder`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/struct.ReedSolomonDecoder.html
[RSD::add_original_shard]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/struct.ReedSolomonDecoder.html#method.add_original_shard
[RSD::add_recovery_shard]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/struct.ReedSolomonDecoder.html#method.add_recovery_shard
[RSD::decode]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/struct.ReedSolomonDecoder.html#method.decode

[`Engine`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/engine/trait.Engine.html
[`Rate`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/rate/trait.Rate.html

[mod:rate]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/rate/index.html

[`reed_solomon_16::encode`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/fn.encode.html
[`reed_solomon_16::decode`]: https://docs.rs/reed-solomon-16/0.1.0/reed_solomon_16/fn.decode.html
