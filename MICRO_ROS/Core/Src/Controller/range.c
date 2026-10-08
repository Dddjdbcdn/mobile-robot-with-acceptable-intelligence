#include "range.h"
#include "main.h"
#include "usart.h"
volatile uint16_t range_mm[3];
volatile uint32_t range_last_valid_ms[3];

#define RANGE_RETRY_INTERVAL_MS 1000U
#define RANGE_RESTART_AGE_MS    1000U

typedef struct
{
    uint8_t byte;
    uint8_t frame[8];
    uint8_t index;
    uint8_t output_index;
} VL53L1X_Receiver;

/* GY-53L1 modules (VL53L1X), 9600 baud, 8-N-1. */
static VL53L1X_Receiver vl53l1x_usart2 = {.output_index = 0};
static VL53L1X_Receiver vl53l1x_uart4 = {.output_index = 2};

static uint8_t tf_byte;
static uint8_t tf_frame[9];
static uint8_t tf_index = 0;
static uint32_t range_last_retry_ms[3];
static const uint8_t vl53l1x_continuous_command[3] = {0xA5, 0x45, 0xEA};

static void Restart_VL53L1X_UART(UART_HandleTypeDef *huart,
                                 VL53L1X_Receiver *receiver)
{
    (void)HAL_UART_AbortReceive(huart);
    receiver->index = 0;
    if (HAL_UART_Receive_IT(huart, &receiver->byte, 1) == HAL_OK) {
        (void)HAL_UART_Transmit(huart,
                                (uint8_t *)vl53l1x_continuous_command,
                                sizeof(vl53l1x_continuous_command), 20);
    }
}

static void Restart_TFmini_UART(void)
{
    (void)HAL_UART_AbortReceive(&huart3);
    tf_index = 0;
    (void)HAL_UART_Receive_IT(&huart3, &tf_byte, 1);
}

void Range_UART_IT_Init(void)
{
    Restart_VL53L1X_UART(&huart2, &vl53l1x_usart2);
    Restart_TFmini_UART();
    Restart_VL53L1X_UART(&huart4, &vl53l1x_uart4);

    uint32_t now = HAL_GetTick();
    range_last_retry_ms[0] = now;
    range_last_retry_ms[1] = now;
    range_last_retry_ms[2] = now;
}

bool Range_IsFresh(uint8_t index, uint32_t now, uint32_t max_age_ms)
{
    return index < 3U && range_last_valid_ms[index] != 0U &&
           (now - range_last_valid_ms[index]) <= max_age_ms;
}

void Range_Service(uint32_t now)
{
    for (uint8_t i = 0; i < 3U; i++) {
        if (Range_IsFresh(i, now, RANGE_RESTART_AGE_MS) ||
            (now - range_last_retry_ms[i]) < RANGE_RETRY_INTERVAL_MS) {
            continue;
        }

        range_last_retry_ms[i] = now;
        if (i == 0U) {
            Restart_VL53L1X_UART(&huart2, &vl53l1x_usart2);
        } else if (i == 1U) {
            Restart_TFmini_UART();
        } else {
            Restart_VL53L1X_UART(&huart4, &vl53l1x_uart4);
        }
    }
}

static void Parse_VL53L1X_Byte(VL53L1X_Receiver *receiver)
{
    uint8_t byte = receiver->byte;

    if (receiver->index == 0)
    {
        if (byte == 0x5A)
            receiver->frame[receiver->index++] = byte;
        return;
    }

    if (receiver->index == 1)
    {
        if (byte == 0x5A)
            receiver->frame[receiver->index++] = byte;
        else
            receiver->index = 0;
        return;
    }

    receiver->frame[receiver->index++] = byte;

    if ((receiver->index == 3 && receiver->frame[2] != 0x15) ||
        (receiver->index == 4 && receiver->frame[3] != 0x03))
    {
        receiver->index = (byte == 0x5A) ? 1 : 0;
        receiver->frame[0] = byte;
        return;
    }

    if (receiver->index == sizeof(receiver->frame))
    {
        uint8_t checksum = 0;

        for (uint8_t i = 0; i < sizeof(receiver->frame) - 1; i++)
            checksum += receiver->frame[i];

        /* The upper nibble of byte 6 is RangeStatus; zero means reliable. */
        if (checksum == receiver->frame[7] &&
            ((receiver->frame[6] >> 4) & 0x0F) == 0)
        {
            uint16_t distance =
                ((uint16_t)receiver->frame[4] << 8) |
                receiver->frame[5];

            if (distance >= 50 && distance <= 4000) {
                range_mm[receiver->output_index] = distance;
                range_last_valid_ms[receiver->output_index] = HAL_GetTick();
            }
        }

        receiver->index = 0;
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
                range_mm[1] = distance * 10U;
                range_last_valid_ms[1] = HAL_GetTick();
            }
        }

        tf_index = 0;
    }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        Parse_VL53L1X_Byte(&vl53l1x_usart2);

        if (HAL_UART_Receive_IT(&huart2, &vl53l1x_usart2.byte, 1) != HAL_OK)
        {
            HAL_UART_AbortReceive(&huart2);
            vl53l1x_usart2.index = 0;
            HAL_UART_Receive_IT(&huart2, &vl53l1x_usart2.byte, 1);
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
        Parse_VL53L1X_Byte(&vl53l1x_uart4);

        if (HAL_UART_Receive_IT(&huart4, &vl53l1x_uart4.byte, 1) != HAL_OK)
        {
            HAL_UART_AbortReceive(&huart4);
            vl53l1x_uart4.index = 0;
            HAL_UART_Receive_IT(&huart4, &vl53l1x_uart4.byte, 1);
        }
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        HAL_UART_AbortReceive(huart);
        vl53l1x_usart2.index = 0;
        HAL_UART_Receive_IT(&huart2, &vl53l1x_usart2.byte, 1);
    }
    else if (huart->Instance == UART4)
    {
        HAL_UART_AbortReceive(huart);
        vl53l1x_uart4.index = 0;
        HAL_UART_Receive_IT(&huart4, &vl53l1x_uart4.byte, 1);
    }
    else if (huart->Instance == USART3)
    {
        HAL_UART_AbortReceive(huart);
        tf_index = 0;
        HAL_UART_Receive_IT(&huart3, &tf_byte, 1);
    }
}
