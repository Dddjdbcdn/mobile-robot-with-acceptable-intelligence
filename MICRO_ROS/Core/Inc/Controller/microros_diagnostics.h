#ifndef MICROROS_DIAGNOSTICS_H
#define MICROROS_DIAGNOSTICS_H

#include "main.h"

#include <stdbool.h>
#include <rcl/rcl.h>
#include <rmw/types.h>

typedef enum {
    PUB_IMU,
    PUB_RANGE,
    PUB_TOF,
    PUB_WHEEL,
    PUB_DEBUG,
    PUB_COUNT
} PublisherDiagnosticId;

typedef enum {
    SUB_WHEEL_CMD,
    SUB_SERVO_TILT,
    SUB_SERVO_PAN,
    SUB_COUNT
} SubscriptionDiagnosticId;

typedef struct {
    rcl_ret_t init_ret;
    rcl_ret_t timer_ret;
    rcl_ret_t executor_ret;
    rcl_ret_t last_publish_ret;
    rcl_ret_t last_publish_error_ret;
    uint32_t callback_count;
    uint32_t publish_success_count;
    uint32_t publish_error_count;
    uint32_t consecutive_publish_error_count;
} PublisherDiagnostics;

typedef struct {
    rcl_ret_t init_ret;
    rcl_ret_t executor_ret;
    uint32_t callback_count;
    uint32_t last_receive_ms;
} SubscriptionDiagnostics;

typedef struct {
    PublisherDiagnostics publishers[PUB_COUNT];
    SubscriptionDiagnostics subscriptions[SUB_COUNT];
    rcl_ret_t support_init_ret;
    rcl_ret_t node_init_ret;
    rcl_ret_t executor_init_ret;
    rcl_ret_t last_executor_error_ret;
    rmw_ret_t last_agent_ping_ret;
    uint32_t executor_error_count;
    uint32_t executor_error_total;
    uint32_t executor_outage_start_ms;
    uint32_t executor_recovery_errors;
    uint32_t executor_recovery_ms;
    uint32_t agent_ping_error_count;
    uint32_t agent_ping_error_total;
    uint32_t agent_outage_start_ms;
    uint32_t agent_recovery_errors;
    uint32_t agent_recovery_ms;
    uint32_t debug_drop_count;
    uint32_t previous_publisher_callbacks[PUB_COUNT];
    uint32_t previous_health_report_ms;
    uint32_t previous_tof_irq_count;
    uint32_t previous_tof_frame_count;
    uint32_t previous_transport_error_total;
    uint32_t previous_rx_restart_attempts;
    uint32_t previous_rx_restart_successes;
    uint32_t previous_rx_restart_failures;
    uint32_t previous_system_report_ms;
    float imu_accel_sum[3];
    float imu_gyro_sum[3];
    float imu_accel_peak_g;
    float imu_gyro_peak_dps;
    uint32_t imu_sample_count;
    bool executor_recovery_pending;
    bool agent_recovery_pending;
    bool debug_ready;
} MicroRosDiagnostics;

void MicroRosDiagnostics_CaptureResetCause(void);
void MicroRosDiagnostics_Reset(MicroRosDiagnostics *diagnostics);
void MicroRosDiagnostics_SetDebugReady(MicroRosDiagnostics *diagnostics,
                                       bool ready);
void MicroRosDiagnostics_RecordSubscription(MicroRosDiagnostics *diagnostics,
                                            SubscriptionDiagnosticId id,
                                            uint32_t now);
void MicroRosDiagnostics_RecordExecutor(MicroRosDiagnostics *diagnostics,
                                        rcl_ret_t result, uint32_t now);
void MicroRosDiagnostics_RecordAgentPing(MicroRosDiagnostics *diagnostics,
                                         rmw_ret_t result, uint32_t now);
void MicroRosDiagnostics_RecordImuSample(MicroRosDiagnostics *diagnostics,
                                         float ax, float ay, float az,
                                         float gx, float gy, float gz);
rcl_ret_t MicroRosDiagnostics_Publish(MicroRosDiagnostics *diagnostics,
                                      PublisherDiagnosticId id,
                                      rcl_publisher_t *publisher,
                                      const void *message);
rcl_ret_t MicroRosDiagnostics_PublishPair(MicroRosDiagnostics *diagnostics,
                                          PublisherDiagnosticId id,
                                          rcl_publisher_t *first_publisher,
                                          const void *first_message,
                                          rcl_publisher_t *second_publisher,
                                          const void *second_message);
void MicroRosDiagnostics_Log(MicroRosDiagnostics *diagnostics,
                             rcl_publisher_t *debug_publisher,
                             const char *format, ...);
void MicroRosDiagnostics_ReportBoot(MicroRosDiagnostics *diagnostics,
                                    rcl_publisher_t *debug_publisher);
void MicroRosDiagnostics_ReportIfDue(MicroRosDiagnostics *diagnostics,
                                     rcl_publisher_t *debug_publisher,
                                     uint32_t now);

#endif
