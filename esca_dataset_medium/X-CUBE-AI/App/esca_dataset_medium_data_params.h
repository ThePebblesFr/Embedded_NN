/**
  ******************************************************************************
  * @file    esca_dataset_medium_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Oct 17 15:37:44 2022
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef ESCA_DATASET_MEDIUM_DATA_PARAMS_H
#define ESCA_DATASET_MEDIUM_DATA_PARAMS_H
#pragma once

#include "ai_platform.h"

/*
#define AI_ESCA_DATASET_MEDIUM_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_esca_dataset_medium_data_weights_params[1]))
*/

#define AI_ESCA_DATASET_MEDIUM_DATA_CONFIG               (NULL)


#define AI_ESCA_DATASET_MEDIUM_DATA_ACTIVATIONS_SIZES \
  { 1933840, }
#define AI_ESCA_DATASET_MEDIUM_DATA_ACTIVATIONS_SIZE     (1933840)
#define AI_ESCA_DATASET_MEDIUM_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ESCA_DATASET_MEDIUM_DATA_ACTIVATION_1_SIZE    (1933840)



#define AI_ESCA_DATASET_MEDIUM_DATA_WEIGHTS_SIZES \
  { 746504, }
#define AI_ESCA_DATASET_MEDIUM_DATA_WEIGHTS_SIZE         (746504)
#define AI_ESCA_DATASET_MEDIUM_DATA_WEIGHTS_COUNT        (1)
#define AI_ESCA_DATASET_MEDIUM_DATA_WEIGHT_1_SIZE        (746504)



#define AI_ESCA_DATASET_MEDIUM_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_esca_dataset_medium_activations_table[1])

extern ai_handle g_esca_dataset_medium_activations_table[1 + 2];



#define AI_ESCA_DATASET_MEDIUM_DATA_WEIGHTS_TABLE_GET() \
  (&g_esca_dataset_medium_weights_table[1])

extern ai_handle g_esca_dataset_medium_weights_table[1 + 2];


#endif    /* ESCA_DATASET_MEDIUM_DATA_PARAMS_H */
