#ifndef RANGE_H
#define RANGE_H

#include "main.h"
#include <stdbool.h>

extern volatile uint16_t range_mm[3];
extern volatile uint32_t range_last_valid_ms[3];

void Range_UART_IT_Init(void);
void Range_Service(uint32_t now);
bool Range_IsFresh(uint8_t index, uint32_t now, uint32_t max_age_ms);

#endif
