#ifndef ULTRASONIC_H
#define ULTRASONIC_H

#include "main.h"

extern volatile uint16_t final_dist[3];

void UART_IT_Init(void);

#endif