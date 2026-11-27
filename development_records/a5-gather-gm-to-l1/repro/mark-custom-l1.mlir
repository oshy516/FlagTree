module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">} {
  func.func @custom_l1() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %dst = memref.alloc() : memref<2x1x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_MTE2>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "custom_gather_gm_to_l1_half"} "gather_gm_to_l1" outs(%dst : memref<2x1x16x16xf16, #hivm.address_space<cbuf>>)
    return
  }
  func.func @copy_control() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %dst = memref.alloc() : memref<2x1x16x16xf16, #hivm.address_space<cbuf>>
    %empty = tensor.empty() : tensor<2x1x16x16xf16>
    hivm.hir.copy ins(%empty : tensor<2x1x16x16xf16>) outs(%dst : memref<2x1x16x16xf16, #hivm.address_space<cbuf>>)
    return
  }
}
