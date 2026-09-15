#include "ultrasonic.h"
#include "main.h"
#include "usart.h"
#include <stdbool.h>

uint8_t cmd_reg = 0x55;
uint8_t rx_data[3][2];
volatile uint16_t final_dist[3];

uint32_t WAIT_TIME = 15;

/* UART2: GY-53L1 module (VL53L1X), 9600 baud, 8-N-1. */
static uint8_t vl53l1x_byte;
static uint8_t vl53l1x_frame[8];
static uint8_t vl53l1x_index = 0;

static uint8_t tf_byte;
static uint8_t tf_frame[9];
static uint8_t tf_index = 0;

void UART_IT_Init(void)
{
    static const uint8_t vl53l1x_continuous_command[3] = {0xA5, 0x45, 0xEA};

    vl53l1x_index = 0;
    HAL_UART_Receive_IT(&huart2, &vl53l1x_byte, 1);
    HAL_UART_Transmit(&huart2, (uint8_t *)vl53l1x_continuous_command,
                      sizeof(vl53l1x_continuous_command), 20);

    HAL_UART_Receive_IT(&huart3, &tf_byte, 1);
}

static void Parse_VL53L1X_Byte(uint8_t byte)
{
    if (vl53l1x_index == 0)
    {
        if (byte == 0x5A)
            vl53l1x_frame[vl53l1x_index++] = byte;
        return;
    }

    if (vl53l1x_index == 1)
    {
        if (byte == 0x5A)
            vl53l1x_frame[vl53l1x_index++] = byte;
        else
            vl53l1x_index = 0;
        return;
    }

    vl53l1x_frame[vl53l1x_index++] = byte;

    if ((vl53l1x_index == 3 && vl53l1x_frame[2] != 0x15) ||
        (vl53l1x_index == 4 && vl53l1x_frame[3] != 0x03))
    {
        vl53l1x_index = (byte == 0x5A) ? 1 : 0;
        vl53l1x_frame[0] = byte;
        return;
    }

    if (vl53l1x_index == sizeof(vl53l1x_frame))
    {
        uint8_t checksum = 0;

        for (uint8_t i = 0; i < sizeof(vl53l1x_frame) - 1; i++)
            checksum += vl53l1x_frame[i];

        /* The upper nibble of byte 6 is RangeStatus; zero means reliable. */
        if (checksum == vl53l1x_frame[7] &&
            ((vl53l1x_frame[6] >> 4) & 0x0F) == 0)
        {
            uint16_t distance =
                ((uint16_t)vl53l1x_frame[4] << 8) |
                vl53l1x_frame[5];

            if (distance >= 50 && distance <= 4000)
                final_dist[0] = distance;
        }

        vl53l1x_index = 0;
    }
}

static void Parse_TFminiS_Byte(uint8_t byte)
{
    if (tf_index == 0)
    {
        if (byte == 0x59)
            tf_frame[tf_index++] = byte;

        return;
    }

    if (tf_index == 1)
    {
        if (byte == 0x59)
            tf_frame[tf_index++] = byte;
        else
            tf_index = 0;

        return;
    }

    tf_frame[tf_index++] = byte;

    if (tf_index == 9)
    {
        uint8_t checksum = 0;

        for (uint8_t i = 0; i < 8; i++)
            checksum += tf_frame[i];

        if (checksum == tf_frame[8])
        {
            uint16_t distance =
                ((uint16_t)tf_frame[3] << 8) |
                tf_frame[2];

            if (distance != 0xFFFF &&
                distance != 0xFFFE &&
                distance != 0xFFFC)
            {
                final_dist[1] = distance * 10.0;
            }
        }

        tf_index = 0;
    }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        Parse_VL53L1X_Byte(vl53l1x_byte);

        if (HAL_UART_Receive_IT(&huart2, &vl53l1x_byte, 1) != HAL_OK)
        {
            HAL_UART_AbortReceive(&huart2);
            vl53l1x_index = 0;
            HAL_UART_Receive_IT(&huart2, &vl53l1x_byte, 1);
        }
    }
    else if (huart->Instance == USART3)
    {
        Parse_TFminiS_Byte(tf_byte);

        if (HAL_UART_Receive_IT(&huart3, &tf_byte, 1) != HAL_OK)
        {
            HAL_UART_AbortReceive(&huart3);
            tf_index = 0;
            HAL_UART_Receive_IT(&huart3, &tf_byte, 1);
        }
    }
    else if (huart->Instance == UART4)
    {
        final_dist[2] =
            ((uint16_t)rx_data[2][0] << 8) |
            rx_data[2][1];
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        HAL_UART_AbortReceive(huart);
        vl53l1x_index = 0;
        HAL_UART_Receive_IT(&huart2, &vl53l1x_byte, 1);
    }
    else if (huart->Instance == USART3)
    {
        HAL_UART_AbortReceive(huart);
        tf_index = 0;
        HAL_UART_Receive_IT(&huart3, &tf_byte, 1);
    }
}

typedef enum
{
    PING_SENSOR_3,
    WAIT_FOR_SENSOR_3,
    ALL_DONE
} SensorStep;

static SensorStep current_step = PING_SENSOR_3;
static uint32_t stopwatch = 0;

void Read_Ultrasonics(void)
{
    switch (current_step)
    {
        case PING_SENSOR_3:
            HAL_UART_AbortReceive_IT(&huart4);
            HAL_UART_Receive_IT(&huart4, rx_data[2], 2);
            HAL_UART_Transmit_IT(&huart4, &cmd_reg, 1);

            stopwatch = HAL_GetTick();
            current_step = WAIT_FOR_SENSOR_3;
            break;

        case WAIT_FOR_SENSOR_3:
            if ((HAL_GetTick() - stopwatch) >= WAIT_TIME)
                current_step = ALL_DONE;
            break;

        case ALL_DONE:
            current_step = PING_SENSOR_3;
            break;
    }
}
