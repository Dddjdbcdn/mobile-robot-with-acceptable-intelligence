#ifndef RANGE_H
#define RANGE_H

#include "main.h"
#include <stdbool.h>

extern volatile uint16_t range_mm[3];
extern volatile uint8_t range_measurement_valid[3];
extern volatile uint32_t range_last_rx_ms[3];
extern volatile uint32_t range_last_response_ms[3];
extern volatile uint32_t range_last_valid_ms[3];
extern volatile uint32_t range_last_rejected_ms[3];
extern volatile uint32_t range_response_frame_count[3];
extern volatile uint32_t range_valid_frame_count[3];
extern volatile uint32_t range_checksum_error_count[3];
extern volatile uint32_t range_rejected_frame_count[3];
extern volatile uint16_t range_last_reject_code[3];
extern volatile uint32_t range_uart_error_count[3];
extern volatile uint32_t range_restart_count[3];

void Range_UART_IT_Init(void);
void Range_Service(uint32_t now);
bool Range_IsFresh(uint8_t index, uint32_t now, uint32_t max_age_ms);
bool Range_IsResponding(uint8_t index, uint32_t now, uint32_t max_age_ms);

#endif
