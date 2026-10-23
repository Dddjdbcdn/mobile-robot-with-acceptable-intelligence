#include "microros_diagnostics.h"

#include "imu.h"
#include "range.h"
#include "vl53l7cx_api.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <std_msgs/msg/string.h>

#define HEALTH_REPORT_PERIOD_MS 1000U
#define IMU_STALE_TIMEOUT_MS 200U
#define RANGE_STALE_TIMEOUT_MS 1000U
#define TOF_STALE_TIMEOUT_MS 500U

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
    return timestamp == 0U ? UINT32_MAX : now - timestamp;
}

static const char *publisher_state(const MicroRosDiagnostics *diagnostics,
                                   PublisherDiagnosticId id,
                                   uint32_t callbacks)
{
    const PublisherDiagnostics *publisher = &diagnostics->publishers[id];

    if (publisher->init_ret != RCL_RET_OK ||
        publisher->timer_ret != RCL_RET_OK ||
        publisher->executor_ret != RCL_RET_OK) {
        return "SETUP_FAIL";
    }
    if (id == PUB_DEBUG) {
        return publisher->last_publish_ret == RCL_RET_OK ? "OK" : "PUBLISH_FAIL";
    }
    if (callbacks == 0U) {
        return "STALLED";
    }
    return publisher->last_publish_ret == RCL_RET_OK ? "OK" : "PUBLISH_FAIL";
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
                                        rcl_ret_t result)
{
    if (result == RCL_RET_OK || result == RCL_RET_TIMEOUT) {
        diagnostics->executor_error_count = 0;
        return;
    }

    diagnostics->last_executor_error_ret = result;
    diagnostics->executor_error_count++;
    diagnostics->executor_error_total++;
}

void MicroRosDiagnostics_RecordAgentPing(MicroRosDiagnostics *diagnostics,
                                         rmw_ret_t result)
{
    diagnostics->last_agent_ping_ret = result;
    if (result == RMW_RET_OK) {
        diagnostics->agent_ping_error_count = 0;
        return;
    }

    diagnostics->agent_ping_error_count++;
    diagnostics->agent_ping_error_total++;
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

static void publish_health_report(MicroRosDiagnostics *diagnostics,
                                  rcl_publisher_t *debug_publisher,
                                  uint32_t now, uint32_t elapsed)
{
    static const char *publisher_names[PUB_COUNT] = {
        "imu", "range", "tof", "pwm", "wheel", "debug"
    };
    static const char *subscription_names[SUB_COUNT] = {
        "wheel_cmd", "servo_tilt", "servo_pan"
    };
    static const char *range_names[3] = {"LEFT", "CENTER", "RIGHT"};
    uint32_t callbacks[PUB_COUNT];
    uint32_t rates[PUB_COUNT];
    bool range_fresh[3];
    bool range_responding[3];
    bool range_rx_active[3];
    uint32_t tof_irq_age = age_ms(now, tof_last_irq_ms);
    uint32_t tof_frame_age = age_ms(now, tof_last_frame_ms);
    uint32_t tof_irq_rate =
        ((tof_irq_count - diagnostics->previous_tof_irq_count) * 1000U) / elapsed;
    uint32_t tof_frame_rate =
        ((tof_frame_count - diagnostics->previous_tof_frame_count) * 1000U) / elapsed;
    bool imu_fresh = mpu_init_status && imu_last_read_ms != 0U &&
        age_ms(now, imu_last_read_ms) <= IMU_STALE_TIMEOUT_MS;
    bool tof_streaming = tof_alive_status && tof_last_frame_ms != 0U &&
        tof_frame_age <= TOF_STALE_TIMEOUT_MS;
    uint8_t tof_valid_zones = 0U;
    uint8_t tof_unfavorable_zones = 0U;
    const char *overall = "OK";
    const char *failing = "none";
    const char *reason = "all_topics_and_sensors_healthy";
    long failure_rc = 0;

    for (uint8_t i = 0; i < PUB_COUNT; i++) {
        callbacks[i] = diagnostics->publishers[i].callback_count -
            diagnostics->previous_publisher_callbacks[i];
        rates[i] = callbacks[i] * 1000U / elapsed;
        diagnostics->previous_publisher_callbacks[i] =
            diagnostics->publishers[i].callback_count;
    }
    for (uint8_t i = 0; i < 3U; i++) {
        range_fresh[i] = Range_IsFresh(i, now, RANGE_STALE_TIMEOUT_MS);
        range_responding[i] = Range_IsResponding(i, now, RANGE_STALE_TIMEOUT_MS);
        range_rx_active[i] = range_last_rx_ms[i] != 0U &&
            age_ms(now, range_last_rx_ms[i]) <= RANGE_STALE_TIMEOUT_MS;
    }
    if (tof_streaming) {
        for (uint8_t i = 0; i < 64U; i++) {
            if (Results.target_status[i] == 5U) {
                tof_valid_zones++;
            } else {
                tof_unfavorable_zones++;
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
        reason = !mpu_init_status ? "i2c_offline" : "sample_stream_stale";
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
               (!range_fresh[0] || !range_fresh[1] || !range_fresh[2])) {
        uint8_t failed = !range_fresh[0] ? 0U : (!range_fresh[1] ? 1U : 2U);
        overall = "DEGRADED";
        failing = failed == 0U ? "range_left" :
            (failed == 1U ? "range_center" : "range_right");
        reason = range_responding[failed] ? "measurement_rejected" :
            (range_rx_active[failed] ? "invalid_uart_frames" : "no_uart_response");
        failure_rc = range_last_reject_code[failed];
    }

    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[HEALTH] overall=%s failing=%s reason=%s rc=%ld uptime_ms=%lu",
        overall, failing, reason, failure_rc, (unsigned long)now);
    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[PUB] imu=%s/%luHz range=%s/%luHz tof=%s/%luHz pwm=%s/%luHz wheel=%s/%luHz",
        publisher_state(diagnostics, PUB_IMU, callbacks[PUB_IMU]), (unsigned long)rates[PUB_IMU],
        publisher_state(diagnostics, PUB_RANGE, callbacks[PUB_RANGE]), (unsigned long)rates[PUB_RANGE],
        publisher_state(diagnostics, PUB_TOF, callbacks[PUB_TOF]), (unsigned long)rates[PUB_TOF],
        publisher_state(diagnostics, PUB_PWM, callbacks[PUB_PWM]), (unsigned long)rates[PUB_PWM],
        publisher_state(diagnostics, PUB_WHEEL, callbacks[PUB_WHEEL]), (unsigned long)rates[PUB_WHEEL]);
    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[PUB_ERRORS] total=%lu,%lu,%lu,%lu,%lu last_rc=%ld,%ld,%ld,%ld,%ld",
        (unsigned long)diagnostics->publishers[PUB_IMU].publish_error_count,
        (unsigned long)diagnostics->publishers[PUB_RANGE].publish_error_count,
        (unsigned long)diagnostics->publishers[PUB_TOF].publish_error_count,
        (unsigned long)diagnostics->publishers[PUB_PWM].publish_error_count,
        (unsigned long)diagnostics->publishers[PUB_WHEEL].publish_error_count,
        (long)diagnostics->publishers[PUB_IMU].last_publish_error_ret,
        (long)diagnostics->publishers[PUB_RANGE].last_publish_error_ret,
        (long)diagnostics->publishers[PUB_TOF].last_publish_error_ret,
        (long)diagnostics->publishers[PUB_PWM].last_publish_error_ret,
        (long)diagnostics->publishers[PUB_WHEEL].last_publish_error_ret);
    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[RUNTIME] executor=%ld/%lu/%lu agent=%ld/%lu/%lu debug_dropped=%lu",
        (long)diagnostics->last_executor_error_ret,
        (unsigned long)diagnostics->executor_error_count,
        (unsigned long)diagnostics->executor_error_total,
        (long)diagnostics->last_agent_ping_ret,
        (unsigned long)diagnostics->agent_ping_error_count,
        (unsigned long)diagnostics->agent_ping_error_total,
        (unsigned long)diagnostics->debug_drop_count);
    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[SENSOR] imu=%s errors=%lu reconnects=%lu tof=%s valid=%u unfavorable=%u errors=%lu",
        imu_fresh ? "OK" : "STALE", (unsigned long)imu_read_error_count,
        (unsigned long)imu_reconnect_count, tof_streaming ? "OK" : "STALE",
        (unsigned int)tof_valid_zones, (unsigned int)tof_unfavorable_zones,
        (unsigned long)tof_read_error_count);
    MicroRosDiagnostics_Log(diagnostics, debug_publisher,
        "[TOF] irq=%luHz frame=%luHz irq_age=%lu frame_age=%lu rc=%u consecutive=%u starts=%lu",
        (unsigned long)tof_irq_rate, (unsigned long)tof_frame_rate,
        (unsigned long)tof_irq_age, (unsigned long)tof_frame_age,
        (unsigned int)tof_last_read_status,
        (unsigned int)tof_consecutive_read_errors,
        (unsigned long)tof_start_attempt_count);

    for (uint8_t i = 0; i < 3U; i++) {
        const char *state = range_fresh[i] ? "OK" :
            (range_responding[i] ? "REJECTED" :
             (range_rx_active[i] ? "INVALID" : "DISCONNECTED"));
        MicroRosDiagnostics_Log(diagnostics, debug_publisher,
            "[RANGE_%s] state=%s mm=%u valid_age=%lu rx_age=%lu reject=%u valid=%lu rejected=%lu crc=%lu uart=%lu restarts=%lu",
            range_names[i], state, (unsigned int)range_mm[i],
            (unsigned long)age_ms(now, range_last_valid_ms[i]),
            (unsigned long)age_ms(now, range_last_rx_ms[i]),
            (unsigned int)range_last_reject_code[i],
            (unsigned long)range_valid_frame_count[i],
            (unsigned long)range_rejected_frame_count[i],
            (unsigned long)range_checksum_error_count[i],
            (unsigned long)range_uart_error_count[i],
            (unsigned long)range_restart_count[i]);
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

    publish_health_report(diagnostics, debug_publisher, now, elapsed);
    diagnostics->previous_health_report_ms = now;
    diagnostics->previous_tof_irq_count = tof_irq_count;
    diagnostics->previous_tof_frame_count = tof_frame_count;
}
