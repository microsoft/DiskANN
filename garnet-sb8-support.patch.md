# Garnet SB8 (Signed Int8) Support — Manual Patch Instructions
#
# These changes add VectorValueType.SB8 and wire it through the Garnet server
# to match the DiskANN Rust FFI changes (VectorValueType::SB8 = 3, VectorQuantType::Q8).
#
# 4 files to edit:
#
# ══════════════════════════════════════════════════════════════════════
# 1. libs/server/Storage/Session/MainStore/VectorStoreOps.cs
# ══════════════════════════════════════════════════════════════════════
#
# In the VectorValueType enum, after the XB8 member, add SB8:
#
#   FIND:
#         /// <summary>
#         /// Bytes (8 bit).
#         /// </summary>
#         XB8,
#     }
#
#   REPLACE WITH:
#         /// <summary>
#         /// Bytes (8 bit).
#         /// </summary>
#         XB8,
#
#         /// <summary>
#         /// Signed bytes (int8, [-128, 127]).
#         /// </summary>
#         SB8,
#     }
#
#
# ══════════════════════════════════════════════════════════════════════
# 2. libs/server/Resp/Vector/RespServerSessionVectors.cs
# ══════════════════════════════════════════════════════════════════════
#
# --- 2a. In NetworkVADD, after the "XB8" parsing block, add an "SB8" block.
#
#   FIND (the XB8 block in VADD, around line 115):
#                 else if (parseState.GetArgSliceByRef(curIx).Span.EqualsUpperCaseSpanIgnoringCase("XB8"u8))
#                 {
#                     curIx++;
#                     if (curIx >= parseState.Count)
#                     {
#                         return AbortWithWrongNumberOfArguments("VADD");
#                     }
#
#                     var asBytes = parseState.GetArgSliceByRef(curIx).Span;
#                     curIx++;
#
#                     valueType = VectorValueType.XB8;
#                     values = asBytes;
#                 }
#
#   REPLACE WITH:
#                 else if (parseState.GetArgSliceByRef(curIx).Span.EqualsUpperCaseSpanIgnoringCase("XB8"u8))
#                 {
#                     curIx++;
#                     if (curIx >= parseState.Count)
#                     {
#                         return AbortWithWrongNumberOfArguments("VADD");
#                     }
#
#                     var asBytes = parseState.GetArgSliceByRef(curIx).Span;
#                     curIx++;
#
#                     valueType = VectorValueType.XB8;
#                     values = asBytes;
#                 }
#                 else if (parseState.GetArgSliceByRef(curIx).Span.EqualsUpperCaseSpanIgnoringCase("SB8"u8))
#                 {
#                     curIx++;
#                     if (curIx >= parseState.Count)
#                     {
#                         return AbortWithWrongNumberOfArguments("VADD");
#                     }
#
#                     var asBytes = parseState.GetArgSliceByRef(curIx).Span;
#                     curIx++;
#
#                     valueType = VectorValueType.SB8;
#                     values = asBytes;
#                 }
#
#
# --- 2b. In the quant guard (around line 349), allow Q8 through:
#
#   FIND:
#                 if (quantType != VectorQuantType.XPreQ8 && quantType != VectorQuantType.NoQuant)
#
#   REPLACE WITH:
#                 if (quantType != VectorQuantType.XPreQ8 && quantType != VectorQuantType.NoQuant && quantType != VectorQuantType.Q8)
#
#
# --- 2c. In NetworkVSIM, after the "XB8" parsing block, add an "SB8" block.
#
#   FIND (the XB8 block in VSIM, around line 502):
#                     else if (kind.Span.EqualsUpperCaseSpanIgnoringCase("XB8"u8))
#                     {
#                         if (curIx >= parseState.Count)
#                         {
#                             return AbortWithWrongNumberOfArguments("VSIM");
#                         }
#
#                         var asBytes = parseState.GetArgSliceByRef(curIx).Span;
#
#                         valueType = VectorValueType.XB8;
#                         values = asBytes;
#                         curIx++;
#                     }
#
#   REPLACE WITH:
#                     else if (kind.Span.EqualsUpperCaseSpanIgnoringCase("XB8"u8))
#                     {
#                         if (curIx >= parseState.Count)
#                         {
#                             return AbortWithWrongNumberOfArguments("VSIM");
#                         }
#
#                         var asBytes = parseState.GetArgSliceByRef(curIx).Span;
#
#                         valueType = VectorValueType.XB8;
#                         values = asBytes;
#                         curIx++;
#                     }
#                     else if (kind.Span.EqualsUpperCaseSpanIgnoringCase("SB8"u8))
#                     {
#                         if (curIx >= parseState.Count)
#                         {
#                             return AbortWithWrongNumberOfArguments("VSIM");
#                         }
#
#                         var asBytes = parseState.GetArgSliceByRef(curIx).Span;
#
#                         valueType = VectorValueType.SB8;
#                         values = asBytes;
#                         curIx++;
#                     }
#
#
# ══════════════════════════════════════════════════════════════════════
# 3. libs/server/Resp/Vector/DiskANNService.cs
# ══════════════════════════════════════════════════════════════════════
#
# --- 3a. In the Insert method, add SB8 arm after XB8 (around line 79):
#
#   FIND:
#             else if (vectorType == VectorValueType.XB8)
#             {
#                 vector_len = vector.Length;
#             }
#             else
#             {
#                 throw new NotImplementedException($"{vectorType}");
#             }
#
#   REPLACE WITH:
#             else if (vectorType == VectorValueType.XB8)
#             {
#                 vector_len = vector.Length;
#             }
#             else if (vectorType == VectorValueType.SB8)
#             {
#                 vector_len = vector.Length;
#             }
#             else
#             {
#                 throw new NotImplementedException($"{vectorType}");
#             }
#
#
# --- 3b. In SearchVector method, same pattern (around line 117):
#
#   FIND:
#             else if (vectorType == VectorValueType.XB8)
#             {
#                 vector_len = vector.Length;
#             }
#             else
#             {
#                 throw new NotImplementedException($"{vectorType}");
#             }
#
#   REPLACE WITH:
#             else if (vectorType == VectorValueType.XB8)
#             {
#                 vector_len = vector.Length;
#             }
#             else if (vectorType == VectorValueType.SB8)
#             {
#                 vector_len = vector.Length;
#             }
#             else
#             {
#                 throw new NotImplementedException($"{vectorType}");
#             }
#
#
# ══════════════════════════════════════════════════════════════════════
# 4. libs/server/Resp/Vector/VectorManager.cs
# ══════════════════════════════════════════════════════════════════════
#
# --- 4a. In CalculateValueDimensions (around line 945), add SB8:
#
#   FIND:
#             else if (valueType == VectorValueType.XB8)
#             {
#                 return (uint)(values.Length);
#             }
#             else
#             {
#                 throw new NotImplementedException($"{valueType}");
#             }
#
#   REPLACE WITH:
#             else if (valueType == VectorValueType.XB8)
#             {
#                 return (uint)(values.Length);
#             }
#             else if (valueType == VectorValueType.SB8)
#             {
#                 return (uint)(values.Length);
#             }
#             else
#             {
#                 throw new NotImplementedException($"{valueType}");
#             }
#
#
# --- 4b. In TryGetEmbedding (around line 920), add Q8 dequant before the
#         throw NotImplementedException:
#
#   FIND:
#                 else if (quantType == VectorQuantType.XPreQ8)
#                 {
#                     for (var i = 0; i < asBytes.Length; i++)
#                     {
#                         into[i] = from[i];
#                     }
#                 }
#                 else
#                 {
#                     // TODO: Handle Q8 and BIN as they are implemented
#                     throw new NotImplementedException($"Unexpected quantization: {quantType}");
#                 }
#
#   REPLACE WITH:
#                 else if (quantType == VectorQuantType.XPreQ8)
#                 {
#                     for (var i = 0; i < asBytes.Length; i++)
#                     {
#                         into[i] = from[i];
#                     }
#                 }
#                 else if (quantType == VectorQuantType.Q8)
#                 {
#                     // Q8 stores signed bytes; dequantize by sign-extending to float
#                     for (var i = 0; i < asBytes.Length; i++)
#                     {
#                         into[i] = (float)(sbyte)from[i];
#                     }
#                 }
#                 else
#                 {
#                     throw new NotImplementedException($"Unexpected quantization: {quantType}");
#                 }
#
# ══════════════════════════════════════════════════════════════════════
# End of patch instructions.
# ══════════════════════════════════════════════════════════════════════
