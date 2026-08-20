/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : app_freertos.c
  * Description        : Code for freertos applications
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "microros_app.h"
#include "imu.h"
#include "motor.h"
#include "ultrasonic.h"
#include "i2c.h"
#include "tim.h"
#include "vl53l7cx_api.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */

VL53L7CX_Configuration Dev;
VL53L7CX_ResultsData Results;
volatile uint8_t tof_data_ready = 0;
volatile uint8_t tof_alive_status = 0;

/* USER CODE END Variables */
/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .priority = (osPriority_t) osPriorityNormal,
  .stack_size = 3000 * 4
};

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */
void Hardware_Task(void *argument);
/* USER CODE END FunctionPrototypes */

void StartDefaultTask(void *argument);

void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/**
  * @brief  FreeRTOS initialization
  * @param  None
  * @retval None
  */
void MX_FREERTOS_Init(void) {
  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  osThreadId_t hardwareTaskHandle;
  const osThreadAttr_t hardwareTask_attributes = {
    .name = "hardwareTask",
    .priority = (osPriority_t) osPriorityHigh,
    .stack_size = 1024 * 4
  };
  hardwareTaskHandle = osThreadNew(Hardware_Task, NULL, &hardwareTask_attributes);
  (void)hardwareTaskHandle;
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

}

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the defaultTask thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN StartDefaultTask */
  /* Infinite loop */
  run_microros_app();
  /* USER CODE END StartDefaultTask */
}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

void Hardware_Task(void *argument)
{
    // --- 1. SENSOR INITIALIZATION PHASE ---
    
    // Hold VL53L7CX in reset to prevent I2C interference
    HAL_GPIO_WritePin(LPN_GPIO_Port, LPN_Pin, GPIO_PIN_RESET);
    osDelay(10); // Let the bus settle

    // Initialize the IMU while the ToF is asleep
    if (MPU6050_Init(&hi2c1)) {
      MPU6050_Calibrate(&hi2c1);
    }

    // Wake up the VL53L7CX
    HAL_GPIO_WritePin(LPN_GPIO_Port, LPN_Pin, GPIO_PIN_SET);
    osDelay(100); // Wait for ToF internal bootloader to finish

    // Configure and initialize the ToF sensor
    Dev.platform.address = 0x52; 
    
    vl53l7cx_is_alive(&Dev, &tof_alive_status);
    
    if(tof_alive_status) {
        vl53l7cx_init(&Dev);
        vl53l7cx_set_resolution(&Dev, VL53L7CX_RESOLUTION_8X8);
        // vl53l7cx_set_ranging_mode(&Dev,VL53L7CX_RANGING_MODE_CONTINUOUS);
        vl53l7cx_set_ranging_frequency_hz(&Dev, 10);
        vl53l7cx_start_ranging(&Dev);

        
    }

    // --- 2. TIMER INITIALIZATION PHASE ---

    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);
    HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_2);
    HAL_TIM_Encoder_Start_IT(&htim2, TIM_CHANNEL_ALL);
    HAL_TIM_Encoder_Start_IT(&htim3, TIM_CHANNEL_ALL);
    HAL_TIM_Base_Start_IT(&htim4);

    UART_IT_Init();
    
    for(;;)
    {
        Read_IMU();
        Read_Ultrasonics();

        if (tof_data_ready) {
            tof_data_ready = 0; 
            vl53l7cx_get_ranging_data(&Dev, &Results);
        }

        osDelay(5); 
    }
}

/* USER CODE END Application */

