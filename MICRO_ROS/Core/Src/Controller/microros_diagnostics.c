#include "microros_diagnostics.h"

#include "imu.h"
#include "microros_dma_transport.h"
#include "range.h"
#include "vl53l7cx_api.h"

#include "FreeRTOS.h"
#include "task.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <std_msgs/msg/string.h>

#define HEALTH_REPORT_PERIOD_MS 2000U
#define IMU_STALE_TIMEOUT_MS 200U
#define RANGE_STALE_TIMEOUT_MS 1000U
#define TOF_STALE_TIMEOUT_MS 500U
#define SYSTEM_REPORT_PERIOD_MS 30000U

/* VL53L1X target-quality statuses which describe the optical measurement,
 * not the health of the sensor, UART, or STM32.  The range publisher may
 * still reject these measurements when their distance is unreliable. */
#define VL53L1X_STATUS_SIGNAL_FAIL 2U
#define VL53L1X_STATUS_OUT_OF_BOUNDS 4U
#define VL53L1X_STATUS_WRAP_TARGET 7U

#define RESET_CAUSE_OPTION_BYTES (1UL << 0)
#define RESET_CAUSE_PIN          (1UL << 1)
#define RESET_CAUSE_BROWNOUT     (1UL << 2)
#define RESET_CAUSE_SOFTWARE     (1UL << 3)
#define RESET_CAUSE_IWDG         (1UL << 4)
#define RESET_CAUSE_WWDG         (1UL << 5)
#define RESET_CAUSE_LOW_POWER    (1UL << 6)

static uint32_t boot_reset_causes;
static uint32_t boot_rcc_csr;

static void capture_reset_flag(uint32_t rcc_flag, uint32_t cause)
{
    if (__HAL_RCC_GET_FLAG(rcc_flag)) {
        boot_reset_causes |= cause;
    }
}

void MicroRosDiagnostics_CaptureResetCause(void)
{
    boot_rcc_csr = RCC->CSR;
    boot_reset_causes = 0U;

    capture_reset_flag(RCC_FLAG_OBLRST, RESET_CAUSE_OPTION_BYTES);
    capture_reset_flag(RCC_FLAG_PINRST, RESET_CAUSE_PIN);
    capture_reset_flag(RCC_FLAG_BORRST, RESET_CAUSE_BROWNOUT);
    capture_reset_flag(RCC_FLAG_SFTRST, RESET_CAUSE_SOFTWARE);
    capture_reset_flag(RCC_FLAG_IWDGRST, RESET_CAUSE_IWDG);
    capture_reset_flag(RCC_FLAG_WWDGRST, RESET_CAUSE_WWDG);
    capture_reset_flag(RCC_FLAG_LPWRRST, RESET_CAUSE_LOW_POWER);

    __HAL_RCC_CLEAR_RESET_FLAGS();
}

extern VL53L7CX_ResultsData Results;
extern volatile uint8_t tof_alive_status;
extern volatile uint32_t tof_irq_count;
extern volatile uint32_t tof_last_irq_ms;
extern volatile uint32_t tof_frame_count;
extern volatile uint32_t tof_last_frame_ms;
extern volatile uint8_t tof_last_read_status;
extern volatile uint32_t tof_start_attempt_count;
extern volatile uint32_t tof_read_error_count;
extern volatile uint8_t tof_consecutive_read_errors;

static uint32_t age_ms(uint32_t now, uint32_t timestamp)
{
    if (timestamp == 0U) {
        return UINT32_MAX;
    }

    uint32_t elapsed = now - timestamp;
    /* A producer may update its timestamp after this task captured `now`. */
    return elapsed > (UINT32_MAX / 2U) ? 0U : elapsed;
}

static bool range_rejection_is_expected(uint8_t index, uint16_t status)
{
    /* Index 1 is the TFmini center sensor and uses different status values. */
    if (index == 1U) {
        return false;
    }

    return status == VL53L1X_STATUS_SIGNAL_FAIL ||
        status == VL53L1X_STATUS_OUT_OF_BOUNDS ||
        status == VL53L1X_STATUS_WRAP_TARGET;
}

void MicroRosDiagnostics_Reset(MicroRosDiagnostics *diagnostics)
{
    memset(diagnostics, 0, sizeof(*diagnostics));
    diagnostics->support_init_ret = RCL_RET_NOT_INIT;
    diagnostics->node_init_ret = RCL_RET_NOT_INIT;
    diagnostics->executor_init_ret = RCL_RET_NOT_INIT;
    diagnostics->last_executor_error_ret = RCL_RET_NOT_INIT;
    diagnostics->last_agent_ping_ret = RMW_RET_OK;

    for (uint8_t i = 0; i < PUB_COUNT; i++) {
        diagnostics->publishers[i].init_ret = RCL_RET_NOT_INIT;
        diagnostics->publishers[i].timer_ret = RCL_RET_NOT_INIT;
        diagnostics->publishers[i].executor_ret = RCL_RET_NOT_INIT;
        diagnostics->publishers[i].last_publish_ret = RCL_RET_NOT_INIT;
        diagnostics->publishers[i].last_publish_error_ret = RCL_RET_NOT_INIT;
    }
    for (uint8_t i = 0; i < SUB_COUNT; i++) {
        diagnostics->subscriptions[i].init_ret = RCL_RET_NOT_INIT;
        diagnostics->subscriptions[i].executor_ret = RCL_RET_NOT_INIT;
    }
}

void MicroRosDiagnostics_SetDebugReady(MicroRosDiagnostics *diagnostics,
                                       bool ready)
{
    diagnostics->debug_ready = ready;
}

void MicroRosDiagnostics_RecordSubscription(MicroRosDiagnostics *diagnostics,
                                            SubscriptionDiagnosticId id,
                                            uint32_t now)
{
    diagnostics->subscriptions[id].callback_count++;
    diagnostics->subscriptions[id].last_receive_ms = now;
}

void MicroRosDiagnostics_RecordExecutor(MicroRosDiagnostics *diagnostics,
                                        rcl_ret_t result, uint32_t now)
{
    if (result == RCL_RET_OK || result == RCL_RET_TIMEOUT) {
        if (diagnostics->executor_error_count != 0U) {
            diagnostics->executor_recovery_errors =
                diagnostics->executor_error_count;
            diagnostics->executor_recovery_ms =
                now - diagnostics->executor_outage_start_ms;
            diagnostics->executor_recovery_pending = true;
        }
        diagnostics->executor_error_count = 0;
        return;
    }

    if (diagnostics->executor_error_count == 0U) {
        diagnostics->executor_outage_start_ms = now;
    }
    diagnostics->last_executor_error_ret = result;
    diagnostics->executor_error_count++;
    diagnostics->executor_error_total++;
}

void MicroRosDiagnostics_RecordAgentPing(MicroRosDiagnostics *diagnostics,
                                         rmw_ret_t result, uint32_t now)
{
    diagnostics->last_agent_ping_ret = result;
    if (result == RMW_RET_OK) {
        if (diagnostics->agent_ping_error_count != 0U) {
            diagnostics->agent_recovery_errors =
                diagnostics->agent_ping_error_count;
            diagnostics->agent_recovery_ms =
                now - diagnostics->agent_outage_start_ms;
            diagnostics->agent_recovery_pending = true;
        }
        diagnostics->agent_ping_error_count = 0;
        return;
    }

    if (diagnostics->agent_ping_error_count == 0U) {
        diagnostics->agent_outage_start_ms = now;
    }
    diagnostics->agent_ping_error_count++;
    diagnostics->agent_ping_error_total++;
}

void MicroRosDiagnostics_RecordImuSample(MicroRosDiagnostics *diagnostics,
                                         float ax, float ay, float az,
                                         float gx, float gy, float gz)
{
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(az) ||
        !isfinite(gx) || !isfinite(gy) || !isfinite(gz)) {
        return;
    }

    diagnostics->imu_accel_sum[0] += ax;
    diagnostics->imu_accel_sum[1] += ay;
    diagnostics->imu_accel_sum[2] += az;
    diagnostics->imu_gyro_sum[0] += gx;
    diagnostics->imu_gyro_sum[1] += gy;
    diagnostics->imu_gyro_sum[2] += gz;
    diagnostics->imu_sample_count++;

    float accel_magnitude = sqrtf(ax * ax + ay * ay + az * az);
    float gyro_magnitude = sqrtf(gx * gx + gy * gy + gz * gz);
    if (accel_magnitude > diagnostics->imu_accel_peak_g) {
        diagnostics->imu_accel_peak_g = accel_magnitude;
    }
    if (gyro_magnitude > diagnostics->imu_gyro_peak_dps) {
        diagnostics->imu_gyro_peak_dps = gyro_magnitude;
    }
}

rcl_ret_t MicroRosDiagnostics_Publish(MicroRosDiagnostics *diagnostics,
                                      PublisherDiagnosticId id,
                                      rcl_publisher_t *publisher,
                                      const void *message)
{
    PublisherDiagnostics *state = &diagnostics->publishers[id];
    state->callback_count++;
    state->last_publish_ret = rcl_publish(publisher, message, NULL);

    if (state->last_publish_ret == RCL_RET_OK) {
        state->publish_success_count++;
        state->consecutive_publish_error_count = 0;
    } else {
        state->last_publish_error_ret = state->last_publish_ret;
        state->publish_error_count++;
        state->consecutive_publish_error_count++;
    }
    return state->last_publish_ret;
}

rcl_ret_t MicroRosDiagnostics_PublishPair(MicroRosDiagnostics *diagnostics,
                                          PublisherDiagnosticId id,
                                          rcl_publisher_t *first_publisher,
                                          const void *first_message,
                                          rcl_publisher_t *second_publisher,
                                          const void *second_message)
{
    PublisherDiagnostics *state = &diagnostics->publishers[id];
    rcl_ret_t first_result = rcl_publish(
        first_publisher, first_message, NULL);
    rcl_ret_t second_result = rcl_publish(
        second_publisher, second_message, NULL);

    state->callback_count++;
    state->last_publish_ret = first_result != RCL_RET_OK ?
        first_result : second_result;
    if (state->last_publish_ret == RCL_RET_OK) {
        state->publish_success_count++;
        state->consecutive_publish_error_count = 0;
    } else {
        state->last_publish_error_ret = state->last_publish_ret;
        state->publish_error_count++;
        state->consecutive_publish_error_count++;
    }
    return state->last_publish_ret;
}

static void publish_log(MicroRosDiagnostics *diagnostics,
                        rcl_publisher_t *debug_publisher,
                        const char *format, va_list args)
{
    static char buffer[256];
    static std_msgs__msg__String message;

    if (!diagnostics->debug_ready) {
        diagnostics->debug_drop_count++;
        return;
    }

    vsnprintf(buffer, sizeof(buffer), format, args);

    message.data.data = buffer;
    message.data.size = strlen(buffer);
    message.data.capacity = sizeof(buffer);
    MicroRosDiagnostics_Publish(
        diagnostics, PUB_DEBUG, debug_publisher, &message);
}

void MicroRosDiagnostics_Log(MicroRosDiagnostics *diagnostics,
                             rcl_publisher_t *debug_publisher,
                             const char *format, ...)
{
    va_list args;
    va_start(args, format);
    publish_log(diagnostics, debug_publisher, format, args);
    va_end(args);
}

static void append_reset_cause(char *buffer, size_t capacity,
                               const char *cause, bool *first)
{
    size_t used = strlen(buffer);
    if (used >= capacity) {
        return;
    }
    (void)snprintf(buffer + used, capacity - used, "%s%s",
                   *first ? "" : "+", cause);
    *first = false;
}

void MicroRosDiagnostics_ReportBoot(MicroRosDiagnostics *diagnostics,
                                    rcl_publisher_t *debug_publisher)
{
    char causes[96] = "";
    bool first = true;

    if (boot_reset_causes & RESET_CAUSE_OPTION_BYTES) {
        append_reset_cause(causes, sizeof(causes), "option_bytes", &first);
    }
    if (boot_reset_causes & RESET_CAUSE_PIN) {
        append_reset_cause(causes, sizeof(causes), "reset_pin", &first);
    }
    if (boot_reset_causes & RESET_CAUSE_BROWNOUT) {
        append_reset_cause(
            causes, sizeof(causes), "brownout_or_power_on", &first);
    }
    if (boot_reset_causes & RESET_CAUSE_SOFTWARE) {
        append_reset_cause(causes, sizeof(causes), "software", &first);
    }
    if (boot_reset_causes & RESET_CAUSE_IWDG) {
        append_reset_cause(
            causes, sizeof(causes), "independent_watchdog", &first);
    }
    if (boot_reset_causes & RESET_CAUSE_WWDG) {
        append_reset_cause(
            causes, sizeof(causes), "window_watchdog", &first);
    }
    if (boot_reset_causes & RESET_CAUSE_LOW_POWER) {
        append_reset_cause(causes, sizeof(causes), "low_power", &first);
    }
    if (first) {
        append_reset_cause(causes, sizeof(causes), "unknown", &first);
    }

    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[BOOT] reset_cause=%s flags=0x%02lx rcc_csr=0x%08lx",
        causes, (unsigned long)boot_reset_causes,
        (unsigned long)boot_rcc_csr);
}

static uint32_t transport_error_total(
    const MicroRosTransportDiagnostics *transport)
{
    return transport->open_failures + transport->close_failures +
        transport->tx_ready_timeouts + transport->tx_complete_timeouts +
        transport->tx_hal_errors + transport->rx_state_errors;
}

static void report_pending_events(MicroRosDiagnostics *diagnostics,
                                  rcl_publisher_t *debug_publisher)
{
    MicroRosTransportDiagnostics transport;
    MicroRosTransport_GetDiagnostics(&transport);
    uint32_t transport_total = transport_error_total(&transport);

    if (diagnostics->agent_recovery_pending) {
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[MICROROS] agent_recovered failed_pings=%lu downtime_ms=%lu total=%lu",
            (unsigned long)diagnostics->agent_recovery_errors,
            (unsigned long)diagnostics->agent_recovery_ms,
            (unsigned long)diagnostics->agent_ping_error_total);
        diagnostics->agent_recovery_pending = false;
    }
    if (diagnostics->executor_recovery_pending) {
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[MICROROS] executor_recovered errors=%lu downtime_ms=%lu total=%lu last_rc=%ld",
            (unsigned long)diagnostics->executor_recovery_errors,
            (unsigned long)diagnostics->executor_recovery_ms,
            (unsigned long)diagnostics->executor_error_total,
            (long)diagnostics->last_executor_error_ret);
        diagnostics->executor_recovery_pending = false;
    }
    if (transport_total != diagnostics->previous_transport_error_total) {
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[TRANSPORT] errors=%lu open=%lu close=%lu tx_ready=%lu tx_timeout=%lu tx_hal=%lu rx_state=%lu uart=0x%lx dma_rx=0x%lx dma_tx=0x%lx",
            (unsigned long)transport_total,
            (unsigned long)transport.open_failures,
            (unsigned long)transport.close_failures,
            (unsigned long)transport.tx_ready_timeouts,
            (unsigned long)transport.tx_complete_timeouts,
            (unsigned long)transport.tx_hal_errors,
            (unsigned long)transport.rx_state_errors,
            (unsigned long)transport.last_uart_error,
            (unsigned long)transport.last_rx_dma_error,
            (unsigned long)transport.last_tx_dma_error);
        diagnostics->previous_transport_error_total = transport_total;
    }
    if (transport.rx_restart_attempts !=
            diagnostics->previous_rx_restart_attempts ||
        transport.rx_restart_successes !=
            diagnostics->previous_rx_restart_successes ||
        transport.rx_restart_failures !=
            diagnostics->previous_rx_restart_failures) {
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[TRANSPORT] rx_recovery losses=%lu callbacks=%lu attempts=%lu ok=%lu fail=%lu state=0x%lx uart=0x%lx dma=0x%lx last_ms=%lu max_ms=%lu",
            (unsigned long)transport.rx_loss_events,
            (unsigned long)transport.rx_error_callbacks,
            (unsigned long)transport.rx_restart_attempts,
            (unsigned long)transport.rx_restart_successes,
            (unsigned long)transport.rx_restart_failures,
            (unsigned long)transport.last_rx_state,
            (unsigned long)transport.last_rx_loss_uart_error,
            (unsigned long)transport.last_rx_loss_dma_error,
            (unsigned long)transport.last_rx_recovery_ms,
            (unsigned long)transport.longest_rx_recovery_ms);
        diagnostics->previous_rx_restart_attempts =
            transport.rx_restart_attempts;
        diagnostics->previous_rx_restart_successes =
            transport.rx_restart_successes;
        diagnostics->previous_rx_restart_failures =
            transport.rx_restart_failures;
    }
}

static void report_system_if_due(MicroRosDiagnostics *diagnostics,
                                 rcl_publisher_t *debug_publisher,
                                 uint32_t now)
{
    if ((now - diagnostics->previous_system_report_ms) <
        SYSTEM_REPORT_PERIOD_MS) {
        return;
    }

    uint32_t publish_errors = 0U;
    for (uint8_t i = 0; i < PUB_COUNT; i++) {
        publish_errors += diagnostics->publishers[i].publish_error_count;
    }

    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[SYSTEM] uptime_ms=%lu heap_free=%lu heap_min=%lu task_stack_min_words=%lu exec_errors=%lu ping_errors=%lu publish_errors=%lu debug_drops=%lu",
        (unsigned long)now,
        (unsigned long)xPortGetFreeHeapSize(),
        (unsigned long)xPortGetMinimumEverFreeHeapSize(),
        (unsigned long)uxTaskGetStackHighWaterMark(NULL),
        (unsigned long)diagnostics->executor_error_total,
        (unsigned long)diagnostics->agent_ping_error_total,
        (unsigned long)publish_errors,
        (unsigned long)diagnostics->debug_drop_count);
    diagnostics->previous_system_report_ms = now;
}

static void report_imu_window(MicroRosDiagnostics *diagnostics,
                              rcl_publisher_t *debug_publisher,
                              uint32_t now)
{
    uint32_t samples = diagnostics->imu_sample_count;
    float range_left_m = Range_IsFresh(0U, now, RANGE_STALE_TIMEOUT_MS) ?
        range_mm[0] / 1000.0f : NAN;
    float range_center_m = Range_IsFresh(1U, now, RANGE_STALE_TIMEOUT_MS) ?
        range_mm[1] / 1000.0f : NAN;
    float range_right_m = Range_IsFresh(2U, now, RANGE_STALE_TIMEOUT_MS) ?
        range_mm[2] / 1000.0f : NAN;

    if (samples == 0U) {
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[SENSOR] imu_samples=0 imu_status=no_fresh_samples range_m=(%.3f,%.3f,%.3f) range_rc=(%u,%u,%u)",
            range_left_m, range_center_m, range_right_m,
            (unsigned int)range_last_status_code[0],
            (unsigned int)range_last_status_code[1],
            (unsigned int)range_last_status_code[2]);
        return;
    }

    float divisor = (float)samples;
    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[SENSOR] imu_samples=%lu accel_avg_g=(%.3f,%.3f,%.3f) accel_peak_g=%.3f gyro_avg_dps=(%.2f,%.2f,%.2f) gyro_peak_dps=%.2f range_m=(%.3f,%.3f,%.3f) range_rc=(%u,%u,%u)",
        (unsigned long)samples,
        diagnostics->imu_accel_sum[0] / divisor,
        diagnostics->imu_accel_sum[1] / divisor,
        diagnostics->imu_accel_sum[2] / divisor,
        diagnostics->imu_accel_peak_g,
        diagnostics->imu_gyro_sum[0] / divisor,
        diagnostics->imu_gyro_sum[1] / divisor,
        diagnostics->imu_gyro_sum[2] / divisor,
        diagnostics->imu_gyro_peak_dps,
        range_left_m, range_center_m, range_right_m,
        (unsigned int)range_last_status_code[0],
        (unsigned int)range_last_status_code[1],
        (unsigned int)range_last_status_code[2]);

    for (uint8_t i = 0; i < 3U; i++) {
        diagnostics->imu_accel_sum[i] = 0.0f;
        diagnostics->imu_gyro_sum[i] = 0.0f;
    }
    diagnostics->imu_accel_peak_g = 0.0f;
    diagnostics->imu_gyro_peak_dps = 0.0f;
    diagnostics->imu_sample_count = 0U;
}

static void publish_health_report(MicroRosDiagnostics *diagnostics,
                                  rcl_publisher_t *debug_publisher,
                                  uint32_t now)
{
    static const char *publisher_names[PUB_COUNT] = {
        "imu", "range", "tof", "wheel", "debug"
    };
    static const char *subscription_names[SUB_COUNT] = {
        "wheel_cmd", "servo_tilt", "servo_pan"
    };
    uint32_t callbacks[PUB_COUNT];
    bool range_fresh[3];
    bool range_responding[3];
    bool range_rx_active[3];
    bool range_healthy[3];
    bool imu_online = mpu_init_status;
    uint32_t imu_sample_ms = imu_last_read_ms;
    bool tof_alive = tof_alive_status != 0U;
    uint32_t tof_irq_ms = tof_last_irq_ms;
    uint32_t tof_frame_ms = tof_last_frame_ms;
    uint32_t tof_irq_age = age_ms(now, tof_irq_ms);
    uint32_t tof_frame_age = age_ms(now, tof_frame_ms);
    bool imu_fresh = imu_online && imu_sample_ms != 0U &&
        age_ms(now, imu_sample_ms) <= IMU_STALE_TIMEOUT_MS;
    bool tof_streaming = tof_alive && tof_frame_ms != 0U &&
        tof_frame_age <= TOF_STALE_TIMEOUT_MS;
    uint8_t tof_valid_zones = 0U;
    const char *overall = "OK";
    const char *failing = "none";
    const char *reason = "all_topics_and_sensors_healthy";
    long failure_rc = 0;

    for (uint8_t i = 0; i < PUB_COUNT; i++) {
        callbacks[i] = diagnostics->publishers[i].callback_count -
            diagnostics->previous_publisher_callbacks[i];
        diagnostics->previous_publisher_callbacks[i] =
            diagnostics->publishers[i].callback_count;
    }
    for (uint8_t i = 0; i < 3U; i++) {
        range_fresh[i] = Range_IsFresh(i, now, RANGE_STALE_TIMEOUT_MS);
        range_responding[i] = Range_IsResponding(i, now, RANGE_STALE_TIMEOUT_MS);
        range_rx_active[i] = range_last_rx_ms[i] != 0U &&
            age_ms(now, range_last_rx_ms[i]) <= RANGE_STALE_TIMEOUT_MS;
        range_healthy[i] = range_fresh[i] ||
            (range_responding[i] &&
             range_rejection_is_expected(i, range_last_reject_code[i]));
    }
    if (tof_streaming) {
        for (uint8_t i = 0; i < 64U; i++) {
            if (Results.target_status[i] == 5U ||
                Results.target_status[i] == 9U) {
                tof_valid_zones++;
            }
        }
    }

    if (diagnostics->support_init_ret != RCL_RET_OK) {
        overall = "FAIL"; failing = "micro_ros"; reason = "support_init";
        failure_rc = diagnostics->support_init_ret;
    } else if (diagnostics->node_init_ret != RCL_RET_OK) {
        overall = "FAIL"; failing = "micro_ros"; reason = "node_init";
        failure_rc = diagnostics->node_init_ret;
    } else if (diagnostics->executor_init_ret != RCL_RET_OK) {
        overall = "FAIL"; failing = "micro_ros"; reason = "executor_init";
        failure_rc = diagnostics->executor_init_ret;
    } else if (diagnostics->executor_error_count != 0U) {
        overall = "FAIL"; failing = "micro_ros"; reason = "executor_spin";
        failure_rc = diagnostics->last_executor_error_ret;
    } else if (diagnostics->last_agent_ping_ret != RMW_RET_OK) {
        overall = "FAIL"; failing = "micro_ros"; reason = "agent_ping";
        failure_rc = diagnostics->last_agent_ping_ret;
    }

    for (uint8_t i = 0; i < PUB_DEBUG && overall[0] != 'F'; i++) {
        const PublisherDiagnostics *publisher = &diagnostics->publishers[i];
        if (publisher->init_ret != RCL_RET_OK ||
            publisher->timer_ret != RCL_RET_OK ||
            publisher->executor_ret != RCL_RET_OK) {
            overall = "FAIL";
            failing = publisher_names[i];
            reason = "setup";
        } else if (callbacks[i] == 0U) {
            overall = "FAIL"; failing = publisher_names[i]; reason = "timer_stalled";
        } else if (publisher->last_publish_ret != RCL_RET_OK) {
            overall = "FAIL"; failing = publisher_names[i]; reason = "publish";
            failure_rc = publisher->last_publish_ret;
        }
    }

    for (uint8_t i = 0; i < SUB_COUNT && overall[0] != 'F'; i++) {
        const SubscriptionDiagnostics *subscription = &diagnostics->subscriptions[i];
        if (subscription->init_ret != RCL_RET_OK ||
            subscription->executor_ret != RCL_RET_OK) {
            overall = "FAIL"; failing = subscription_names[i]; reason = "setup";
        }
    }

    if (overall[0] != 'F' && !imu_fresh) {
        overall = "FAIL"; failing = "imu_sensor";
        reason = !imu_online ? "i2c_offline" : "sample_stream_stale";
        failure_rc = imu_consecutive_read_errors;
    } else if (overall[0] != 'F' && !tof_streaming) {
        overall = "FAIL"; failing = "tof_sensor";
        reason = tof_irq_age > TOF_STALE_TIMEOUT_MS ?
            "interrupt_stream_stopped" : "frame_stream_stalled";
        failure_rc = tof_last_read_status;
    } else if (overall[0] != 'F' && tof_last_read_status != VL53L7CX_STATUS_OK) {
        overall = "DEGRADED"; failing = "tof_sensor";
        reason = "transient_frame_read_error";
        failure_rc = tof_last_read_status;
    } else if (overall[0] != 'F' && tof_valid_zones == 0U) {
        overall = "DEGRADED"; failing = "tof_sensor"; reason = "no_valid_targets";
    } else if (overall[0] != 'F' &&
               (!range_healthy[0] || !range_healthy[1] || !range_healthy[2])) {
        uint8_t failed = !range_healthy[0] ? 0U :
            (!range_healthy[1] ? 1U : 2U);
        overall = "DEGRADED";
        failing = failed == 0U ? "range_left" :
            (failed == 1U ? "range_center" : "range_right");
        reason = range_responding[failed] ? "measurement_rejected" :
            (range_rx_active[failed] ? "invalid_uart_frames" : "no_uart_response");
        failure_rc = range_last_reject_code[failed];
    }

    if (overall[0] == 'O') {
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[HEALTH] OK");
    } else {
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[HEALTH] %s failing=%s reason=%s rc=%ld",
            overall, failing, reason, failure_rc);
    }
}

void MicroRosDiagnostics_ReportIfDue(MicroRosDiagnostics *diagnostics,
                                     rcl_publisher_t *debug_publisher,
                                     uint32_t now)
{
    uint32_t elapsed = now - diagnostics->previous_health_report_ms;
    if (elapsed < HEALTH_REPORT_PERIOD_MS) {
        return;
    }

    report_pending_events(diagnostics, debug_publisher);
    report_system_if_due(diagnostics, debug_publisher, now);
    report_imu_window(diagnostics, debug_publisher, now);
    publish_health_report(diagnostics, debug_publisher, now);
    diagnostics->previous_health_report_ms = now;
    diagnostics->previous_tof_irq_count = tof_irq_count;
    diagnostics->previous_tof_frame_count = tof_frame_count;
}
