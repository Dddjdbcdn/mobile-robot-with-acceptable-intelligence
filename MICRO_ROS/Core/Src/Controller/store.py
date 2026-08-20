#include "ultrasonic.h"
#include "microros_app.h"
#include "main.h"
#include "usart.h"

uint8_t cmd_reg = 0x55;
uint8_t rx_data[3][2];
volatile uint16_t final_dist[3];
uint32_t WAIT_TIME = 15;

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART2) {
        final_dist[0] = (rx_data[0][0] << 8) | rx_data[0][1];
    }
    else if (huart->Instance == USART3) {
        final_dist[1] = (rx_data[1][0] << 8) | rx_data[1][1];
    }
    else if (huart->Instance == UART4) {
        final_dist[2] = (rx_data[2][0] << 8) | rx_data[2][1];
    }
}

typedef enum {
    PING_SENSOR_1,
    WAIT_FOR_SENSOR_1,
    PING_SENSOR_2,
    WAIT_FOR_SENSOR_2,
    PING_SENSOR_3,
    WAIT_FOR_SENSOR_3,
    ALL_DONE
} SensorStep;

SensorStep current_step = PING_SENSOR_1;
uint32_t stopwatch = 0;                 

void Read_Ultrasonics(void) {
    switch (current_step) {
        
        case PING_SENSOR_1:
            // 1. Send the ping signal to Sensor 1
            HAL_UART_AbortReceive_IT(&huart2);
            HAL_UART_Receive_IT(&huart2, rx_data[0], 2);
            HAL_UART_Transmit(&huart2, &cmd_reg, 1, 10);
            
            // 2. Look at our watch and move to the next step
            stopwatch = HAL_GetTick(); 
            current_step = WAIT_FOR_SENSOR_1;
            break;

        case WAIT_FOR_SENSOR_1:
            // Has it been 40ms? If yes, move to Sensor 2. If no, do nothing!
            if (HAL_GetTick() - stopwatch >= WAIT_TIME) {
                current_step = PING_SENSOR_2;
            }
            break;

        case PING_SENSOR_2:
            // 1. Send the ping signal to Sensor 2
            HAL_UART_AbortReceive_IT(&huart3);
            HAL_UART_Receive_IT(&huart3, rx_data[1], 2);
            HAL_UART_Transmit(&huart3, &cmd_reg, 1, 10);
            
            // 2. Look at our watch again
            stopwatch = HAL_GetTick();
            current_step = WAIT_FOR_SENSOR_2;
            break;

        case WAIT_FOR_SENSOR_2:
            // Has it been 40ms?
            if (HAL_GetTick() - stopwatch >= WAIT_TIME) {
                current_step = PING_SENSOR_3;
            }
            break;

        case PING_SENSOR_3:
            // 1. Send the ping signal to Sensor 3
            HAL_UART_AbortReceive_IT(&huart4);
            HAL_UART_Receive_IT(&huart4, rx_data[2], 2);
            HAL_UART_Transmit(&huart4, &cmd_reg, 1, 10);
            
            // 2. Look at our watch again
            stopwatch = HAL_GetTick();
            current_step = WAIT_FOR_SENSOR_3;
            break;

        case WAIT_FOR_SENSOR_3:
             // Has it been 40ms?
            if (HAL_GetTick() - stopwatch >= WAIT_TIME) {
                current_step = ALL_DONE; 
            }
            break;

        case ALL_DONE:
            // Reset the checklist to start over next time
            current_step = PING_SENSOR_1; 
            return true; // "Hey Main Loop, I finally have all 3 distances!"
    }

    return false; // "Hey Main Loop, I'm still waiting, ask me again later."
}