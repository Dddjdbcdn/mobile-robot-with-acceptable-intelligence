#ifndef MICROROS_DMA_TRANSPORT_H
#define MICROROS_DMA_TRANSPORT_H

#include <stdint.h>

typedef struct {
    uint32_t open_failures;
    uint32_t close_failures;
    uint32_t tx_ready_timeouts;
    uint32_t tx_complete_timeouts;
    uint32_t tx_hal_errors;
    uint32_t rx_state_errors;
    uint32_t rx_loss_events;
    uint32_t rx_error_callbacks;
    uint32_t rx_restart_attempts;
    uint32_t rx_restart_successes;
    uint32_t rx_restart_failures;
    uint32_t last_rx_state;
    uint32_t last_rx_loss_uart_error;
    uint32_t last_rx_loss_dma_error;
    uint32_t last_rx_recovery_ms;
    uint32_t longest_rx_recovery_ms;
    uint32_t last_uart_error;
    uint32_t last_rx_dma_error;
    uint32_t last_tx_dma_error;
} MicroRosTransportDiagnostics;

/* Safe to call from HAL_UART_ErrorCallback; recovery runs in task context. */
void MicroRosTransport_RequestRxRestart(void);
void MicroRosTransport_GetDiagnostics(
    MicroRosTransportDiagnostics *diagnostics);

#endif
