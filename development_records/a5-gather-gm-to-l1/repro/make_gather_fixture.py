#!/usr/bin/env python3
"""Write a synthetic Gather plus transposed-B matmul adapter for offline tests."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dtype", choices=("fp16", "bf16"))
parser.add_argument("bitcode", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--producer-syntax", action="store_true")
parser.add_argument("--tile", type=int, choices=(16, 32), default=16)
parser.add_argument("--cols", type=int, choices=(32, 64), default=32)
args = parser.parse_args()
element, suffix = ("f16", "half") if args.dtype == "fp16" else ("bf16", "bf16")
bitcode = json.dumps(str(args.bitcode.resolve(strict=True)))
n, d = args.tile, args.cols
tile = f"tensor<{n}x{d}x{element}>"
source = f"memref<{n}x{d}x{element}, strided<[{d}, 1]>>"
indices = f"memref<{n}x1xi32, strided<[1, 1]>>"
result = f"tensor<{n}x{n}xf32>"
result_type = f" to {tile}" if args.producer_syntax else ""
args.output.write_text(f"""module attributes {{hacc.target = #hacc.target<"Ascend950DT_9582">}} {{
  func.func @gather_gm_to_l1_dot_kernel(
      %lock: memref<?xi8>, %workspace: memref<?xi8>,
      %rows: memref<?x{element}>, %indices: memref<?xi32>,
      %query: memref<?x{element}> {{tt.tensor_kind = 0 : i32}},
      %result: memref<?xf32> {{tt.tensor_kind = 1 : i32}},
      %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32)
      attributes {{SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                  global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"}} {{
    %tile_size = arith.constant {n} : i32
    %cols = arith.constant {d} : i32
    %zero = arith.constant 0.0 : f32
    %row_view = memref.reinterpret_cast %rows to offset: [0], sizes: [{n}, {d}], strides: [{d}, 1]
        : memref<?x{element}> to {source}
    %index_view = memref.reinterpret_cast %indices to offset: [0], sizes: [{n}, 1], strides: [1, 1]
        : memref<?xi32> to {indices}
    %scratch = tensor.empty() : {tile}
    %gather = hivm.hir.custom {{bitcode = {bitcode}, hivm.pipe = #hivm.pipe<PIPE_MTE2>,
                                hivm.tcore_type = #hivm.tcore_type<CUBE>,
                                symbol = "custom_gather_gm_to_l1_{suffix}"}}
        "gather_gm_to_l1" ins(%row_view, %index_view, %tile_size, %cols : {source}, {indices}, i32, i32)
        outs(%scratch : {tile}) -> {tile}
    %query_view = memref.reinterpret_cast %query to offset: [0], sizes: [{n}, {d}], strides: [{d}, 1]
        : memref<?x{element}> to {source}
    %query_copy = memref.alloc() : memref<{n}x{d}x{element}>
    memref.copy %query_view, %query_copy : {source} to memref<{n}x{d}x{element}>
    %lhs = bufferization.to_tensor %query_copy restrict writable : memref<{n}x{d}x{element}>{result_type}
    %acc = tensor.empty() : {result}
    %init = linalg.fill ins(%zero : f32) outs(%acc : {result}) -> {result}
    %transpose_init = tensor.empty() : tensor<{d}x{n}x{element}>
    %rhs = linalg.transpose ins(%gather : {tile}) outs(%transpose_init : tensor<{d}x{n}x{element}>)
        permutation = [1, 0]
    %dot = linalg.matmul
        {{indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                          affine_map<(i, j, k) -> (k, j)>,
                          affine_map<(i, j, k) -> (i, j)>], input_precision = "ieee"}}
        ins(%lhs, %rhs : {tile}, tensor<{d}x{n}x{element}>) outs(%init : {result}) -> {result}
    %output = memref.reinterpret_cast %result to offset: [0], sizes: [{n}, {n}], strides: [{n}, 1]
        : memref<?xf32> to memref<{n}x{n}xf32, strided<[{n}, 1]>>
    bufferization.materialize_in_destination %dot in writable %output
        : ({result}, memref<{n}x{n}xf32, strided<[{n}, 1]>>) -> ()
    return
  }}
}}
""", encoding="utf-8")
