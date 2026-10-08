#include <uxr/client/transport.h>

#include <rmw_microxrcedds_c/config.h>

#include "main.h"
#include "cmsis_os.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef RMW_UXRCE_TRANSPORT_CUSTOM

#define UART_DMA_BUFFER_SIZE 2048U
#define UART_TX_READY_TIMEOUT_MS 20U
#define UART_TX_COMPLETE_TIMEOUT_MS 100U

static uint8_t dma_buffer[UART_DMA_BUFFER_SIZE];
static size_t dma_head = 0;
static size_t dma_tail = 0;

bool cubemx_transport_open(struct uxrCustomTransport *transport)
{
    UART_HandleTypeDef *uart = (UART_HandleTypeDef *)transport->args;

    /* DMA restarts at offset zero, so software indices must restart too. */
    (void)HAL_UART_DMAStop(uart);
    dma_head = 0;
    dma_tail = 0;

    return HAL_UART_Receive_DMA(uart, dma_buffer, UART_DMA_BUFFER_SIZE) == HAL_OK;
}

bool cubemx_transport_close(struct uxrCustomTransport *transport)
{
    UART_HandleTypeDef *uart = (UART_HandleTypeDef *)transport->args;
    HAL_StatusTypeDef status = HAL_UART_DMAStop(uart);

    dma_head = 0;
    dma_tail = 0;
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
            status = HAL_TIMEOUT;
            break;
        }
        osDelay(1);
    }

    if (status != HAL_OK && err != NULL) {
        *err = 1;
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
    if (uart->RxState != HAL_UART_STATE_BUSY_RX) {
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
