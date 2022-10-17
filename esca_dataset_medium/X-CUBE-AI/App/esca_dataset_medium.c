/**
  ******************************************************************************
  * @file    esca_dataset_medium.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Oct 17 15:37:44 2022
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "esca_dataset_medium.h"
#include "esca_dataset_medium_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_esca_dataset_medium
 
#undef AI_ESCA_DATASET_MEDIUM_MODEL_SIGNATURE
#define AI_ESCA_DATASET_MEDIUM_MODEL_SIGNATURE     "ae581571eabbdcb124ac518e30dff09a"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Mon Oct 17 15:37:44 2022"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_ESCA_DATASET_MEDIUM_N_BATCHES
#define AI_ESCA_DATASET_MEDIUM_N_BATCHES         (1)

static ai_ptr g_esca_dataset_medium_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_esca_dataset_medium_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 172800, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 460800, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 115200, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56320, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 14080, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1600, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  activation_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  activation_6_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 864, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_conv2d_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 11520, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_conv2d_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5760, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_conv2d_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5760, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_conv2d_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2816, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_conv2d_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 704, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 180, 320), AI_STRIDE_INIT(4, 4, 4, 12, 2160),
  1, &input_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_conv2d_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 90, 160), AI_STRIDE_INIT(4, 4, 4, 128, 11520),
  1, &conv2d_conv2d_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_conv2d_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 45, 80), AI_STRIDE_INIT(4, 4, 4, 128, 5760),
  1, &conv2d_1_conv2d_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_conv2d_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 22, 40), AI_STRIDE_INIT(4, 4, 4, 256, 5632),
  1, &conv2d_2_conv2d_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_conv2d_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 11, 20), AI_STRIDE_INIT(4, 4, 4, 256, 2816),
  1, &conv2d_3_conv2d_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 5, 10), AI_STRIDE_INIT(4, 4, 4, 128, 640),
  1, &conv2d_4_conv2d_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_output0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 1600, 1, 1), AI_STRIDE_INIT(4, 4, 4, 6400, 6400),
  1, &conv2d_4_conv2d_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_dense_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  activation_5_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &activation_5_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &dense_1_dense_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  activation_6_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &activation_6_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_conv2d_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 3, 3, 3, 32), AI_STRIDE_INIT(4, 4, 12, 36, 108),
  1, &conv2d_conv2d_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_conv2d_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_conv2d_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_conv2d_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 384, 1152),
  1, &conv2d_1_conv2d_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_conv2d_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_1_conv2d_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_conv2d_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 4, 128, 384, 1152),
  1, &conv2d_2_conv2d_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_conv2d_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_2_conv2d_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_conv2d_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 768, 2304),
  1, &conv2d_3_conv2d_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_conv2d_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_3_conv2d_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_weights, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 32), AI_STRIDE_INIT(4, 4, 256, 768, 2304),
  1, &conv2d_4_conv2d_weights_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_4_conv2d_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1600, 64, 1, 1), AI_STRIDE_INIT(4, 4, 6400, 409600, 409600),
  1, &dense_dense_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_dense_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 64, 2, 1, 1), AI_STRIDE_INIT(4, 4, 256, 512, 512),
  1, &dense_1_dense_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &dense_1_dense_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_conv2d_scratch0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 180, 2), AI_STRIDE_INIT(4, 4, 4, 128, 23040),
  1, &conv2d_conv2d_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_conv2d_scratch0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 90, 2), AI_STRIDE_INIT(4, 4, 4, 128, 11520),
  1, &conv2d_1_conv2d_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_conv2d_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 45, 2), AI_STRIDE_INIT(4, 4, 4, 256, 11520),
  1, &conv2d_2_conv2d_scratch0_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_conv2d_scratch0, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 22, 2), AI_STRIDE_INIT(4, 4, 4, 256, 5632),
  1, &conv2d_3_conv2d_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_scratch0, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 11, 2), AI_STRIDE_INIT(4, 4, 4, 128, 1408),
  1, &conv2d_4_conv2d_scratch0_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_6_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &activation_6_chain,
  NULL, &activation_6_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_1_dense_weights, &dense_1_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_dense_layer, 19,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_1_dense_chain,
  NULL, &activation_6_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_5_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_5_chain,
  NULL, &dense_1_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_conv2d_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_dense_weights, &dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_dense_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_dense_chain,
  NULL, &activation_5_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_conv2d_weights, &conv2d_4_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_conv2d_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_conv2d_layer, 14,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &conv2d_4_conv2d_chain,
  NULL, &dense_dense_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_conv2d_weights, &conv2d_3_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_conv2d_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_conv2d_layer, 11,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &conv2d_3_conv2d_chain,
  NULL, &conv2d_4_conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_conv2d_weights, &conv2d_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_conv2d_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_conv2d_layer, 8,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &conv2d_2_conv2d_chain,
  NULL, &conv2d_3_conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_conv2d_weights, &conv2d_1_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_conv2d_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_conv2d_layer, 5,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &conv2d_1_conv2d_chain,
  NULL, &conv2d_2_conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_conv2d_weights, &conv2d_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_conv2d_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_conv2d_layer, 2,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &conv2d_conv2d_chain,
  NULL, &conv2d_1_conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 746504, 1, 1),
    746504, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1933840, 1, 1),
    1933840, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ESCA_DATASET_MEDIUM_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ESCA_DATASET_MEDIUM_OUT_NUM, &activation_6_output),
  &conv2d_conv2d_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 746504, 1, 1),
      746504, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1933840, 1, 1),
      1933840, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ESCA_DATASET_MEDIUM_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ESCA_DATASET_MEDIUM_OUT_NUM, &activation_6_output),
  &conv2d_conv2d_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool esca_dataset_medium_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_esca_dataset_medium_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_0_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 1196560);
    input_0_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 1196560);
    
    conv2d_conv2d_scratch0_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 1887760);
    conv2d_conv2d_scratch0_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 1887760);
    
    conv2d_conv2d_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 11776);
    conv2d_conv2d_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 11776);
    
    conv2d_1_conv2d_scratch0_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 1854976);
    conv2d_1_conv2d_scratch0_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 1854976);
    
    conv2d_1_conv2d_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    conv2d_1_conv2d_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    
    conv2d_2_conv2d_scratch0_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 460800);
    conv2d_2_conv2d_scratch0_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 460800);
    
    conv2d_2_conv2d_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 483840);
    conv2d_2_conv2d_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 483840);
    
    conv2d_3_conv2d_scratch0_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    conv2d_3_conv2d_scratch0_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    
    conv2d_3_conv2d_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 11264);
    conv2d_3_conv2d_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 11264);
    
    conv2d_4_conv2d_scratch0_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    conv2d_4_conv2d_scratch0_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    
    conv2d_4_conv2d_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 2816);
    conv2d_4_conv2d_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 2816);
    
    dense_dense_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    dense_dense_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    
    activation_5_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 256);
    activation_5_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 256);
    
    dense_1_dense_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    dense_1_dense_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 0);
    
    activation_6_output_array.data = AI_PTR(g_esca_dataset_medium_activations_map[0] + 8);
    activation_6_output_array.data_start = AI_PTR(g_esca_dataset_medium_activations_map[0] + 8);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool esca_dataset_medium_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_esca_dataset_medium_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_conv2d_weights_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 0);
    conv2d_conv2d_weights_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 0);
    
    conv2d_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_conv2d_bias_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 3456);
    conv2d_conv2d_bias_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 3456);
    
    conv2d_1_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_conv2d_weights_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 3584);
    conv2d_1_conv2d_weights_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 3584);
    
    conv2d_1_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_conv2d_bias_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 40448);
    conv2d_1_conv2d_bias_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 40448);
    
    conv2d_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_conv2d_weights_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 40576);
    conv2d_2_conv2d_weights_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 40576);
    
    conv2d_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_conv2d_bias_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 114304);
    conv2d_2_conv2d_bias_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 114304);
    
    conv2d_3_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_conv2d_weights_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 114560);
    conv2d_3_conv2d_weights_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 114560);
    
    conv2d_3_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_conv2d_bias_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 262016);
    conv2d_3_conv2d_bias_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 262016);
    
    conv2d_4_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_conv2d_weights_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 262272);
    conv2d_4_conv2d_weights_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 262272);
    
    conv2d_4_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_conv2d_bias_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 336000);
    conv2d_4_conv2d_bias_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 336000);
    
    dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_weights_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 336128);
    dense_dense_weights_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 336128);
    
    dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_bias_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 745728);
    dense_dense_bias_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 745728);
    
    dense_1_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_weights_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 745984);
    dense_1_dense_weights_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 745984);
    
    dense_1_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_bias_array.data = AI_PTR(g_esca_dataset_medium_weights_map[0] + 746496);
    dense_1_dense_bias_array.data_start = AI_PTR(g_esca_dataset_medium_weights_map[0] + 746496);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_esca_dataset_medium_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ESCA_DATASET_MEDIUM_MODEL_NAME,
      .model_signature   = AI_ESCA_DATASET_MEDIUM_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 290620032,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_esca_dataset_medium_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ESCA_DATASET_MEDIUM_MODEL_NAME,
      .model_signature   = AI_ESCA_DATASET_MEDIUM_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 290620032,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_esca_dataset_medium_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_esca_dataset_medium_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_esca_dataset_medium_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_esca_dataset_medium_create(network, AI_ESCA_DATASET_MEDIUM_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_esca_dataset_medium_data_params_get(&params) != true) {
        err = ai_esca_dataset_medium_get_error(*network);
        return err;
    }
#if defined(AI_ESCA_DATASET_MEDIUM_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_ESCA_DATASET_MEDIUM_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_esca_dataset_medium_init(*network, &params) != true) {
        err = ai_esca_dataset_medium_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_esca_dataset_medium_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_esca_dataset_medium_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_esca_dataset_medium_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_esca_dataset_medium_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= esca_dataset_medium_configure_weights(net_ctx, params);
  ok &= esca_dataset_medium_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_esca_dataset_medium_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_esca_dataset_medium_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_ESCA_DATASET_MEDIUM_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

