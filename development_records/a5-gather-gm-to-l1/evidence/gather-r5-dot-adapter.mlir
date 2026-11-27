#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">, ssbuffer.insertionOptimization} {
  func.func @dot_baseline(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 32], strides: [32, 1] : memref<?xf16> to memref<16x32xf16, strided<[32, 1]>>
    %alloc = memref.alloc() : memref<16x32xf16>
    memref.copy %reinterpret_cast, %alloc : memref<16x32xf16, strided<[32, 1]>> to memref<16x32xf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf16> to memref<32x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<32x16xf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x16xf16, strided<[16, 1]>> to memref<32x16xf16>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x16xf16>
    %4 = linalg.matmul {indexing_maps = [#map, #map1, #map2], input_precision = "ieee"} ins(%2, %3 : tensor<16x32xf16>, tensor<32x16xf16>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<16x16xf32>, memref<16x16xf32, strided<[16, 1]>>) -> ()
    return
  }
}
