#include <uxr/client/transport.h>

#include <rmw_microxrcedds_c/config.h>

#include "main.h"
#include "cmsis_os.h"
#include "microros_dma_transport.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef RMW_UXRCE_TRANSPORT_CUSTOM

#define UART_DMA_BUFFER_SIZE 2048U
#define UART_TX_READY_TIMEOUT_MS 20U
#define UART_TX_COMPLETE_TIMEOUT_MS 100U
#define UART_RX_RESTART_COOLDOWN_MS 100U

static uint8_t dma_buffer[UART_DMA_BUFFER_SIZE];
static size_t dma_head = 0;
static size_t dma_tail = 0;
static UART_HandleTypeDef *transport_uart = NULL;
static volatile uint32_t open_failures = 0;
static volatile uint32_t close_failures = 0;
static volatile uint32_t tx_ready_timeouts = 0;
static volatile uint32_t tx_complete_timeouts = 0;
static volatile uint32_t tx_hal_errors = 0;
static volatile uint32_t rx_state_errors = 0;
static volatile bool rx_restart_requested = false;
static volatile bool rx_outage_active = false;
static volatile uint32_t rx_loss_events = 0;
static volatile uint32_t rx_error_callbacks = 0;
static volatile uint32_t rx_restart_attempts = 0;
static volatile uint32_t rx_restart_successes = 0;
static volatile uint32_t rx_restart_failures = 0;
static volatile uint32_t rx_outage_started_ms = 0;
static volatile uint32_t last_rx_restart_attempt_ms = 0;
static volatile uint32_t last_rx_state = 0;
static volatile uint32_t last_rx_uart_error = 0;
static volatile uint32_t last_rx_dma_error = 0;
static volatile uint32_t last_rx_recovery_ms = 0;
static volatile uint32_t longest_rx_recovery_ms = 0;

static uint32_t dma_error(const DMA_HandleTypeDef *dma)
{
    return dma == NULL ? 0U : dma->ErrorCode;
}

static void begin_rx_outage(UART_HandleTypeDef *uart)
{
    if (!rx_outage_active) {
        rx_outage_active = true;
        rx_outage_started_ms = HAL_GetTick();
        rx_loss_events++;
        last_rx_state = uart == NULL ? 0U : (uint32_t)uart->RxState;
        last_rx_uart_error = uart == NULL ? 0U : uart->ErrorCode;
        last_rx_dma_error = uart == NULL ? 0U : dma_error(uart->hdmarx);
    }
    rx_restart_requested = true;
}

void MicroRosTransport_RequestRxRestart(void)
{
    rx_error_callbacks++;
    rx_restart_requested = true;
}

static void reset_rx_buffer_positions(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    dma_head = 0;
    dma_tail = 0;
    if (primask == 0U) {
        __enable_irq();
    }
}

static bool restart_rx_dma(UART_HandleTypeDef *uart)
{
    uint32_t now = HAL_GetTick();
    if (rx_restart_attempts != 0U &&
        (now - last_rx_restart_attempt_ms) < UART_RX_RESTART_COOLDOWN_MS) {
        return false;
    }

    last_rx_restart_attempt_ms = now;
    rx_restart_attempts++;

    /* Abort only receive so an active debug/topic transmission is preserved. */
    if (HAL_UART_AbortReceive(uart) != HAL_OK) {
        rx_restart_failures++;
        return false;
    }

    /* Circular DMA restarts at offset zero; discard any partial XRCE frame. */
    reset_rx_buffer_positions();
    if (HAL_UART_Receive_DMA(uart, dma_buffer, UART_DMA_BUFFER_SIZE) != HAL_OK ||
        uart->RxState != HAL_UART_STATE_BUSY_RX) {
        rx_restart_failures++;
        return false;
    }

    now = HAL_GetTick();
    last_rx_recovery_ms = now - rx_outage_started_ms;
    if (last_rx_recovery_ms > longest_rx_recovery_ms) {
        longest_rx_recovery_ms = last_rx_recovery_ms;
    }
    rx_restart_successes++;
    rx_restart_requested = false;
    rx_outage_active = false;
    return true;
}

void MicroRosTransport_GetDiagnostics(
    MicroRosTransportDiagnostics *diagnostics)
{
    if (diagnostics == NULL) {
        return;
    }

    diagnostics->open_failures = open_failures;
    diagnostics->close_failures = close_failures;
    diagnostics->tx_ready_timeouts = tx_ready_timeouts;
    diagnostics->tx_complete_timeouts = tx_complete_timeouts;
    diagnostics->tx_hal_errors = tx_hal_errors;
    diagnostics->rx_state_errors = rx_state_errors;
    diagnostics->rx_loss_events = rx_loss_events;
    diagnostics->rx_error_callbacks = rx_error_callbacks;
    diagnostics->rx_restart_attempts = rx_restart_attempts;
    diagnostics->rx_restart_successes = rx_restart_successes;
    diagnostics->rx_restart_failures = rx_restart_failures;
    diagnostics->last_rx_state = last_rx_state;
    diagnostics->last_rx_loss_uart_error = last_rx_uart_error;
    diagnostics->last_rx_loss_dma_error = last_rx_dma_error;
    diagnostics->last_rx_recovery_ms = last_rx_recovery_ms;
    diagnostics->longest_rx_recovery_ms = longest_rx_recovery_ms;
    diagnostics->last_uart_error =
        transport_uart == NULL ? 0U : transport_uart->ErrorCode;
    diagnostics->last_rx_dma_error = transport_uart == NULL ? 0U :
        dma_error(transport_uart->hdmarx);
    diagnostics->last_tx_dma_error = transport_uart == NULL ? 0U :
        dma_error(transport_uart->hdmatx);
}

bool cubemx_transport_open(struct uxrCustomTransport *transport)
{
    UART_HandleTypeDef *uart = (UART_HandleTypeDef *)transport->args;
    transport_uart = uart;

    /* DMA restarts at offset zero, so software indices must restart too. */
    (void)HAL_UART_DMAStop(uart);
    reset_rx_buffer_positions();
    rx_restart_requested = false;
    rx_outage_active = false;

    bool opened =
        HAL_UART_Receive_DMA(uart, dma_buffer, UART_DMA_BUFFER_SIZE) == HAL_OK;
    if (!opened) {
        open_failures++;
    }
    return opened;
}

bool cubemx_transport_close(struct uxrCustomTransport *transport)
{
    UART_HandleTypeDef *uart = (UART_HandleTypeDef *)transport->args;
    HAL_StatusTypeDef status = HAL_UART_DMAStop(uart);

    reset_rx_buffer_positions();
    rx_restart_requested = false;
    rx_outage_active = false;
    if (status != HAL_OK) {
        close_failures++;
    }
    return status == HAL_OK;
}

size_t cubemx_transport_write(struct uxrCustomTransport *transport,
                              const uint8_t *buf, size_t len, uint8_t *err)
{
    UART_HandleTypeDef *uart = (UART_HandleTypeDef *)transport->args;
    uint32_t wait_started = HAL_GetTick();

    if (err != NULL) {
        *err = 0;
    }

    while (uart->gState != HAL_UART_STATE_READY &&
           (HAL_GetTick() - wait_started) < UART_TX_READY_TIMEOUT_MS) {
        osDelay(1);
    }

    if (uart->gState != HAL_UART_STATE_READY) {
        tx_ready_timeouts++;
        if (err != NULL) {
            *err = 1;
        }
        return 0;
    }

    HAL_StatusTypeDef status =
        HAL_UART_Transmit_DMA(uart, (uint8_t *)buf, len);
    wait_started = HAL_GetTick();

    while (status == HAL_OK && uart->gState != HAL_UART_STATE_READY) {
        if ((HAL_GetTick() - wait_started) >= UART_TX_COMPLETE_TIMEOUT_MS) {
            (void)HAL_UART_AbortTransmit(uart);
            tx_complete_timeouts++;
            status = HAL_TIMEOUT;
            break;
        }
        osDelay(1);
    }

    if (status != HAL_OK) {
        if (status != HAL_TIMEOUT) {
            tx_hal_errors++;
        }
        if (err != NULL) {
            *err = 1;
        }
    }
    return status == HAL_OK ? len : 0;
}

size_t cubemx_transport_read(struct uxrCustomTransport *transport,
                             uint8_t *buf, size_t len, int timeout,
                             uint8_t *err)
{
    UART_HandleTypeDef *uart = (UART_HandleTypeDef *)transport->args;
    int ms_used = 0;
    size_t wrote = 0;

    if (err != NULL) {
        *err = 0;
    }
    if (rx_restart_requested ||
        uart->RxState != HAL_UART_STATE_BUSY_RX) {
        if (!rx_outage_active) {
            if (uart->RxState != HAL_UART_STATE_BUSY_RX) {
                rx_state_errors++;
            }
            begin_rx_outage(uart);
        }
        if (restart_rx_dma(uart)) {
            return 0;
        }
        if (err != NULL) {
            *err = 1;
        }
        return 0;
    }

    do {
        __disable_irq();
        dma_tail = UART_DMA_BUFFER_SIZE - __HAL_DMA_GET_COUNTER(uart->hdmarx);
        __enable_irq();
        if (dma_head == dma_tail) {
            ms_used++;
            osDelay(1);
        }
    } while (dma_head == dma_tail && ms_used < timeout);

    while (dma_head != dma_tail && wrote < len) {
        buf[wrote] = dma_buffer[dma_head];
        dma_head = (dma_head + 1U) % UART_DMA_BUFFER_SIZE;
        wrote++;
    }

    return wrote;
}

#endif /* RMW_UXRCE_TRANSPORT_CUSTOM */
