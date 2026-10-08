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
#include "range.h"
#include "i2c.h"
#include "tim.h"
#include "vl53l7cx_api.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define TOF_RETRY_INTERVAL_MS       2000U
#define TOF_FIRST_FRAME_TIMEOUT_MS  1000U
#define TOF_STALE_TIMEOUT_MS         500U
#define TOF_MAX_READ_ERRORS            3U
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
volatile uint32_t tof_irq_count = 0;
volatile uint32_t tof_last_irq_ms = 0;
volatile uint32_t tof_irq_period_ms = 0;
volatile uint32_t tof_frame_count = 0;
volatile uint32_t tof_last_frame_ms = 0;
volatile uint8_t tof_last_read_status = 0;

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

static bool ToF_Start(void)
{
    uint8_t is_alive = 0;
    uint8_t status;

    tof_alive_status = 0;
    tof_data_ready = 0;
    HAL_GPIO_WritePin(LPN_GPIO_Port, LPN_Pin, GPIO_PIN_RESET);
    osDelay(10);
    HAL_GPIO_WritePin(LPN_GPIO_Port, LPN_Pin, GPIO_PIN_SET);
    osDelay(100);

    Dev.platform.address = 0x52;
    status = vl53l7cx_is_alive(&Dev, &is_alive);
    if (status != VL53L7CX_STATUS_OK || !is_alive)
        goto failed;

    status = vl53l7cx_init(&Dev);
    if (status != VL53L7CX_STATUS_OK)
        goto failed;
    status = vl53l7cx_set_resolution(&Dev, VL53L7CX_RESOLUTION_8X8);
    if (status != VL53L7CX_STATUS_OK)
        goto failed;
    status = vl53l7cx_set_ranging_frequency_hz(&Dev, 10);
    if (status != VL53L7CX_STATUS_OK)
        goto failed;
    status = vl53l7cx_start_ranging(&Dev);
    if (status != VL53L7CX_STATUS_OK)
        goto failed;

    tof_last_read_status = VL53L7CX_STATUS_OK;
    tof_last_irq_ms = 0;
    tof_last_frame_ms = 0;
    tof_alive_status = 1;
    return true;

failed:
    tof_last_read_status = status;
    HAL_GPIO_WritePin(LPN_GPIO_Port, LPN_Pin, GPIO_PIN_RESET);
    return false;
}

void Hardware_Task(void *argument)
{
    uint32_t tof_last_retry_ms;
    uint32_t tof_started_ms = 0;
    uint8_t tof_read_errors = 0;

    // --- 1. SENSOR INITIALIZATION PHASE ---
    
    // Hold VL53L7CX in reset to prevent I2C interference
    HAL_GPIO_WritePin(LPN_GPIO_Port, LPN_Pin, GPIO_PIN_RESET);
    osDelay(10); // Let the bus settle

    // Initialize the IMU while the ToF is asleep
    if (MPU6050_Init(&hi2c1)) {
      (void)MPU6050_Calibrate(&hi2c1);
    }

    if (ToF_Start())
        tof_started_ms = HAL_GetTick();
    tof_last_retry_ms = HAL_GetTick();

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

    Range_UART_IT_Init();
    
    for(;;)
    {
        uint32_t now = HAL_GetTick();

        Read_IMU();
        Range_Service(now);

        if (tof_alive_status && tof_data_ready) {
            tof_data_ready = 0;
            tof_last_read_status = vl53l7cx_get_ranging_data(&Dev, &Results);

            if (tof_last_read_status == VL53L7CX_STATUS_OK) {
                tof_read_errors = 0;
                tof_frame_count++;
                tof_last_frame_ms = now;
            } else if (++tof_read_errors >= TOF_MAX_READ_ERRORS) {
                tof_alive_status = 0;
            }
        }

        bool tof_stale = tof_alive_status &&
            ((tof_last_frame_ms == 0U &&
              (now - tof_started_ms) > TOF_FIRST_FRAME_TIMEOUT_MS) ||
             (tof_last_frame_ms != 0U &&
              (now - tof_last_frame_ms) > TOF_STALE_TIMEOUT_MS));

        if (tof_stale) {
            tof_alive_status = 0;
            tof_last_read_status = VL53L7CX_STATUS_TIMEOUT_ERROR;
        }

        if (!tof_alive_status &&
            (now - tof_last_retry_ms) >= TOF_RETRY_INTERVAL_MS) {
            tof_last_retry_ms = now;
            tof_read_errors = 0;
            if (ToF_Start())
                tof_started_ms = HAL_GetTick();
        }

        osDelay(5);
    }
}

/* USER CODE END Application */

