#include "main.h"
#include "cmsis_os.h"
#include "dma.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

#include "microros_app.h"
#include "imu.h"
#include "motor.h"
#include "range.h"
#include "vl53l7cx_api.h"
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <uxr/client/transport.h>
#include <rmw_microxrcedds_c/config.h>
#include <rmw_microros/rmw_microros.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <sensor_msgs/msg/joint_state.h>
#include <std_msgs/msg/string.h>
#include <std_msgs/msg/float32.h>
#include <geometry_msgs/msg/point32.h>
#include <std_msgs/msg/int16_multi_array.h>

extern void I2C1_Clear_Busy_Flag(void);
extern void MX_I2C1_Init(void);

extern VL53L7CX_ResultsData Results;
extern uint8_t tof_alive_status;
extern uint8_t tof_data_ready;
extern volatile uint32_t tof_irq_count;
extern volatile uint32_t tof_last_irq_ms;
extern volatile uint32_t tof_irq_period_ms;
extern volatile uint32_t tof_frame_count;
extern volatile uint32_t tof_last_frame_ms;
extern volatile uint8_t tof_last_read_status;

#define PI 3.14159265358979323846f
#define WHEEL_RADIUS 0.0325
#define WHEEL_SEPERATION 0.396

// --- TRANSPORT DECLARATIONS ---
bool cubemx_transport_open(struct uxrCustomTransport * transport);
bool cubemx_transport_close(struct uxrCustomTransport * transport);
size_t cubemx_transport_write(struct uxrCustomTransport* transport, const uint8_t * buf, size_t len, uint8_t * err);
size_t cubemx_transport_read(struct uxrCustomTransport* transport, uint8_t* buf, size_t len, int timeout, uint8_t* err);

void * microros_allocate(size_t size, void * state);
void microros_deallocate(void * pointer, void * state);
void * microros_reallocate(void * pointer, size_t size, void * state);
void * microros_zero_allocate(size_t number_of_elements, size_t size_of_element, void * state);

// ==========================================
// GLOBAL MICRO-ROS VARIABLES 
// ==========================================
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rclc_executor_t executor;

// --- PUBLISHERS ---
rcl_publisher_t debug_publisher;
std_msgs__msg__String debug_msg;

rcl_publisher_t imu_publisher;
geometry_msgs__msg__Point32 imu_msg;

rcl_publisher_t range_publisher;
geometry_msgs__msg__Point32 range_msg;

rcl_publisher_t wheel_state_publisher;
sensor_msgs__msg__JointState wheel_state_msg;

rcl_publisher_t pwm_publisher;
geometry_msgs__msg__Point32 pwm_msg;

rcl_publisher_t tof_raw_publisher;
std_msgs__msg__Int16MultiArray tof_raw_msg;

// Publisher Memory Buffers
rosidl_runtime_c__String pub_names[2];
double pub_positions[2];
double pub_velocities[2];
char pub_name_L[] = "left_wheel_joint";
char pub_name_R[] = "right_wheel_joint";

int16_t allocated_data_buffer[64];

// --- SUBSCRIBER ---
rcl_subscription_t wheel_cmd_subscriber;
sensor_msgs__msg__JointState wheel_cmd_msg;

// PWM Command Subscribers
rcl_subscription_t servo_tilt_subscriber;
std_msgs__msg__Float32 servo_tilt_msg;

rcl_subscription_t servo_pan_subscriber;
std_msgs__msg__Float32 servo_pan_msg;

// Subscriber Memory Buffers
rosidl_runtime_c__String sub_names[2];
char sub_name_string_L[30];
char sub_name_string_R[30];
double sub_positions[2];
double sub_velocities[2];
double sub_efforts[2];

// --- TIMERS ---
// --- Initialize Timers ---
rcl_timer_t wheel_timer;
rcl_timer_t imu_timer;
rcl_timer_t range_timer;
rcl_timer_t tof_timer;
rcl_timer_t pwm_timer;

typedef enum {
    PUB_IMU,
    PUB_RANGE,
    PUB_TOF,
    PUB_PWM,
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
    uint32_t callback_count;
    uint32_t publish_success_count;
    uint32_t publish_error_count;
} PublisherDiagnostics;

typedef struct {
    rcl_ret_t init_ret;
    rcl_ret_t executor_ret;
    uint32_t callback_count;
    uint32_t last_receive_ms;
} SubscriptionDiagnostics;

static PublisherDiagnostics publisher_diag[PUB_COUNT];
static SubscriptionDiagnostics subscription_diag[SUB_COUNT];
static rcl_ret_t support_init_ret = RCL_RET_NOT_INIT;
static rcl_ret_t node_init_ret = RCL_RET_NOT_INIT;
static rcl_ret_t executor_init_ret = RCL_RET_NOT_INIT;
static rcl_ret_t last_executor_spin_ret = RCL_RET_NOT_INIT;
static uint32_t executor_spin_error_count = 0;
static uint32_t debug_drop_count = 0;
static bool debug_publisher_ready = false;

// ==========================================
// CALLBACKS & FUNCTIONS
// ==========================================

void servo_tilt_callback(const void * msgin)
{
    const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
    subscription_diag[SUB_SERVO_TILT].callback_count++;
    subscription_diag[SUB_SERVO_TILT].last_receive_ms = HAL_GetTick();
    Servo_SetTiltTarget(msg->data);
}

void servo_pan_callback(const void * msgin)
{
    const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
    subscription_diag[SUB_SERVO_PAN].callback_count++;
    subscription_diag[SUB_SERVO_PAN].last_receive_ms = HAL_GetTick();
    Servo_SetPanTarget(msg->data);
}

uint8_t count = 0;
void debug_print(const char *format, ...)
{
        static char buffer[256];

          if (!debug_publisher_ready) {
              debug_drop_count++;
              return;
          }

          va_list args;
          va_start(args, format);
          vsnprintf(buffer, sizeof(buffer), format, args);
          va_end(args);

          debug_msg.data.data = buffer;
          debug_msg.data.size = strlen(buffer);
          debug_msg.data.capacity = sizeof(buffer);

          publisher_diag[PUB_DEBUG].callback_count++;
          publisher_diag[PUB_DEBUG].last_publish_ret =
              rcl_publish(&debug_publisher, &debug_msg, NULL);
          if (publisher_diag[PUB_DEBUG].last_publish_ret == RCL_RET_OK) {
              publisher_diag[PUB_DEBUG].publish_success_count++;
          } else {
              publisher_diag[PUB_DEBUG].publish_error_count++;
          }

}

static void record_publish(PublisherDiagnosticId id, rcl_publisher_t *publisher,
                           const void *message)
{
    PublisherDiagnostics *diag = &publisher_diag[id];
    diag->callback_count++;
    diag->last_publish_ret = rcl_publish(publisher, message, NULL);
    if (diag->last_publish_ret == RCL_RET_OK) {
        diag->publish_success_count++;
    } else {
        diag->publish_error_count++;
    }
}

static const char *publisher_state(PublisherDiagnosticId id,
                                   uint32_t callbacks_since_report)
{
    const PublisherDiagnostics *diag = &publisher_diag[id];

    if (diag->init_ret != RCL_RET_OK || diag->timer_ret != RCL_RET_OK ||
        diag->executor_ret != RCL_RET_OK) {
        return "SETUP_FAIL";
    }
    if (id == PUB_DEBUG) {
        return (diag->last_publish_ret == RCL_RET_OK) ? "OK" : "PUBLISH_FAIL";
    }
    if (callbacks_since_report == 0U) {
        return "STALLED";
    }
    if (diag->last_publish_ret != RCL_RET_OK) {
        return "PUBLISH_FAIL";
    }
    return "OK";
}

void wheel_cmd_callback(const void * msgin)
{
    const sensor_msgs__msg__JointState * msg = (const sensor_msgs__msg__JointState *)msgin;

    subscription_diag[SUB_WHEEL_CMD].callback_count++;
    subscription_diag[SUB_WHEEL_CMD].last_receive_ms = HAL_GetTick();

    if (msg->velocity.size >= 2)
    {
        float target_rad_s_L = (float)msg->velocity.data[0];
        float target_rad_s_R = (float)msg->velocity.data[1];

        target_rpm_L = target_rad_s_L * (60.0f / (2.0f * PI));
        target_rpm_R = target_rad_s_R * (60.0f / (2.0f * PI));
    }
}

void wheel_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      wheel_state_msg.position.data[0] = (position_L == 0.0f) ? 1e-6 : (double)position_L;
      wheel_state_msg.position.data[1] = (position_R == 0.0f) ? 1e-6 : (double)position_R;
      wheel_state_msg.velocity.data[0] = (double)(motor_rpm_L * 2.0f * PI / 60.0f);
      wheel_state_msg.velocity.data[1] = (double)(motor_rpm_R * 2.0f * PI / 60.0f);

      record_publish(PUB_WHEEL, &wheel_state_publisher, &wheel_state_msg);
    }
}

void imu_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      imu_msg.x = Ax;
      imu_msg.y = Ay;
      imu_msg.z = Gz;
      record_publish(PUB_IMU, &imu_publisher, &imu_msg);
    }
}

void range_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      range_msg.x = range_mm[0] / 1000.0f;
      range_msg.y = range_mm[1] / 1000.0f;
      range_msg.z = range_mm[2] / 1000.0f;
      record_publish(PUB_RANGE, &range_publisher, &range_msg);
    }
}

static void publish_health_report(uint32_t now, uint32_t elapsed,
                                  uint32_t tof_irq_rate,
                                  uint32_t tof_frame_rate,
                                  uint32_t tof_irq_age,
                                  uint32_t tof_frame_age)
{
    static uint32_t previous_callbacks[PUB_COUNT] = {0};
    static const char *publisher_names[PUB_COUNT] = {
        "imu", "range", "tof", "pwm", "wheel", "debug"
    };
    uint32_t callback_delta[PUB_COUNT];
    uint32_t rate_hz[PUB_COUNT];
    const char *overall = "OK";
    const char *failing = "none";
    const char *reason = "all_topics_and_sensors_healthy";
    long failure_rc = 0;

    for (uint8_t i = 0; i < PUB_COUNT; i++) {
        callback_delta[i] = publisher_diag[i].callback_count - previous_callbacks[i];
        rate_hz[i] = (elapsed == 0U) ? 0U :
            (callback_delta[i] * 1000U) / elapsed;
        previous_callbacks[i] = publisher_diag[i].callback_count;
    }

    if (support_init_ret != RCL_RET_OK) {
        overall = "FAIL"; failing = "micro_ros"; reason = "support_init";
        failure_rc = (long)support_init_ret;
    } else if (node_init_ret != RCL_RET_OK) {
        overall = "FAIL"; failing = "micro_ros"; reason = "node_init";
        failure_rc = (long)node_init_ret;
    } else if (executor_init_ret != RCL_RET_OK) {
        overall = "FAIL"; failing = "micro_ros"; reason = "executor_init";
        failure_rc = (long)executor_init_ret;
    } else if (executor_spin_error_count != 0U) {
        overall = "FAIL"; failing = "micro_ros"; reason = "executor_spin";
        failure_rc = (long)last_executor_spin_ret;
    }

    for (uint8_t i = 0; i < PUB_DEBUG && overall[0] != 'F'; i++) {
        PublisherDiagnostics *diag = &publisher_diag[i];
        if (diag->init_ret != RCL_RET_OK) {
            overall = "FAIL"; failing = publisher_names[i];
            reason = "publisher_init"; failure_rc = (long)diag->init_ret;
        } else if (diag->timer_ret != RCL_RET_OK) {
            overall = "FAIL"; failing = publisher_names[i];
            reason = "timer_init"; failure_rc = (long)diag->timer_ret;
        } else if (diag->executor_ret != RCL_RET_OK) {
            overall = "FAIL"; failing = publisher_names[i];
            reason = "executor_add"; failure_rc = (long)diag->executor_ret;
        } else if (callback_delta[i] == 0U) {
            overall = "FAIL"; failing = publisher_names[i];
            reason = "timer_not_firing"; failure_rc = 0;
        } else if (diag->last_publish_ret != RCL_RET_OK) {
            overall = "FAIL"; failing = publisher_names[i];
            reason = "publish"; failure_rc = (long)diag->last_publish_ret;
        }
    }

    for (uint8_t i = 0; i < SUB_COUNT && overall[0] != 'F'; i++) {
        if (subscription_diag[i].init_ret != RCL_RET_OK ||
            subscription_diag[i].executor_ret != RCL_RET_OK) {
            static const char *subscription_names[SUB_COUNT] = {
                "wheel_cmd", "servo_tilt", "servo_pan"
            };
            overall = "FAIL";
            failing = subscription_names[i];
            reason = (subscription_diag[i].init_ret != RCL_RET_OK) ?
                "subscription_init" : "executor_add";
            failure_rc = (long)((subscription_diag[i].init_ret != RCL_RET_OK) ?
                subscription_diag[i].init_ret : subscription_diag[i].executor_ret);
        }
    }

    if (overall[0] != 'F' && !mpu_init_status) {
        overall = "FAIL"; failing = "imu_sensor"; reason = "i2c_offline";
    } else if (overall[0] != 'F' &&
               (!tof_alive_status || tof_last_read_status != VL53L7CX_STATUS_OK ||
                tof_irq_rate == 0U || tof_frame_rate == 0U || tof_frame_age > 500U)) {
        overall = "FAIL"; failing = "tof_sensor";
        reason = !tof_alive_status ? "not_detected" :
            ((tof_last_read_status != VL53L7CX_STATUS_OK) ? "read_error" :
             ((tof_irq_rate == 0U) ? "no_interrupts" : "stale_frames"));
        failure_rc = (long)tof_last_read_status;
    } else if (overall[0] != 'F' &&
               (range_mm[0] == 0U || range_mm[1] == 0U || range_mm[2] == 0U)) {
        overall = "DEGRADED"; failing = "range_sensors";
        reason = "one_or_more_uart_sensors_have_no_valid_data";
    }

    debug_print("[HEALTH] overall=%s failing=%s reason=%s rc=%ld uptime_ms=%lu",
                overall, failing, reason, failure_rc, (unsigned long)now);
    debug_print("[PUB] imu=%s/%luHz range=%s/%luHz tof=%s/%luHz pwm=%s/%luHz wheel=%s/%luHz debug=%s/%luHz",
                publisher_state(PUB_IMU, callback_delta[PUB_IMU]), (unsigned long)rate_hz[PUB_IMU],
                publisher_state(PUB_RANGE, callback_delta[PUB_RANGE]), (unsigned long)rate_hz[PUB_RANGE],
                publisher_state(PUB_TOF, callback_delta[PUB_TOF]), (unsigned long)rate_hz[PUB_TOF],
                publisher_state(PUB_PWM, callback_delta[PUB_PWM]), (unsigned long)rate_hz[PUB_PWM],
                publisher_state(PUB_WHEEL, callback_delta[PUB_WHEEL]), (unsigned long)rate_hz[PUB_WHEEL],
                publisher_state(PUB_DEBUG, callback_delta[PUB_DEBUG]), (unsigned long)rate_hz[PUB_DEBUG]);
    debug_print("[PUB_RC] imu=%ld range=%ld tof=%ld pwm=%ld wheel=%ld debug=%ld errors=%lu,%lu,%lu,%lu,%lu,%lu",
                (long)publisher_diag[PUB_IMU].last_publish_ret,
                (long)publisher_diag[PUB_RANGE].last_publish_ret,
                (long)publisher_diag[PUB_TOF].last_publish_ret,
                (long)publisher_diag[PUB_PWM].last_publish_ret,
                (long)publisher_diag[PUB_WHEEL].last_publish_ret,
                (long)publisher_diag[PUB_DEBUG].last_publish_ret,
                (unsigned long)publisher_diag[PUB_IMU].publish_error_count,
                (unsigned long)publisher_diag[PUB_RANGE].publish_error_count,
                (unsigned long)publisher_diag[PUB_TOF].publish_error_count,
                (unsigned long)publisher_diag[PUB_PWM].publish_error_count,
                (unsigned long)publisher_diag[PUB_WHEEL].publish_error_count,
                (unsigned long)publisher_diag[PUB_DEBUG].publish_error_count);
    debug_print("[INIT_PUB] core=%ld,%ld,%ld imu=%ld,%ld,%ld range=%ld,%ld,%ld tof=%ld,%ld,%ld pwm=%ld,%ld,%ld wheel=%ld,%ld,%ld",
                (long)support_init_ret, (long)node_init_ret, (long)executor_init_ret,
                (long)publisher_diag[PUB_IMU].init_ret, (long)publisher_diag[PUB_IMU].timer_ret, (long)publisher_diag[PUB_IMU].executor_ret,
                (long)publisher_diag[PUB_RANGE].init_ret, (long)publisher_diag[PUB_RANGE].timer_ret, (long)publisher_diag[PUB_RANGE].executor_ret,
                (long)publisher_diag[PUB_TOF].init_ret, (long)publisher_diag[PUB_TOF].timer_ret, (long)publisher_diag[PUB_TOF].executor_ret,
                (long)publisher_diag[PUB_PWM].init_ret, (long)publisher_diag[PUB_PWM].timer_ret, (long)publisher_diag[PUB_PWM].executor_ret,
                (long)publisher_diag[PUB_WHEEL].init_ret, (long)publisher_diag[PUB_WHEEL].timer_ret, (long)publisher_diag[PUB_WHEEL].executor_ret);
    debug_print("[SUB] wheel_cmd=%s/rx%lu servo_tilt=%s/rx%lu servo_pan=%s/rx%lu debug_dropped=%lu",
                (subscription_diag[SUB_WHEEL_CMD].init_ret == RCL_RET_OK && subscription_diag[SUB_WHEEL_CMD].executor_ret == RCL_RET_OK) ? "READY" : "FAIL",
                (unsigned long)subscription_diag[SUB_WHEEL_CMD].callback_count,
                (subscription_diag[SUB_SERVO_TILT].init_ret == RCL_RET_OK && subscription_diag[SUB_SERVO_TILT].executor_ret == RCL_RET_OK) ? "READY" : "FAIL",
                (unsigned long)subscription_diag[SUB_SERVO_TILT].callback_count,
                (subscription_diag[SUB_SERVO_PAN].init_ret == RCL_RET_OK && subscription_diag[SUB_SERVO_PAN].executor_ret == RCL_RET_OK) ? "READY" : "FAIL",
                (unsigned long)subscription_diag[SUB_SERVO_PAN].callback_count,
                (unsigned long)debug_drop_count);
    debug_print("[SENSOR] imu=%s/cal=%u tof=%s/irq%luHz/frame%luHz/irq_age%lu/frame_age%lu/read%u range_mm=%u,%u,%u",
                mpu_init_status ? "OK" : "I2C_FAIL", (unsigned int)mpu_calibration_done,
                (tof_alive_status && tof_last_read_status == VL53L7CX_STATUS_OK && tof_frame_age <= 500U) ? "OK" : "FAIL",
                (unsigned long)tof_irq_rate, (unsigned long)tof_frame_rate,
                (unsigned long)tof_irq_age, (unsigned long)tof_frame_age,
                (unsigned int)tof_last_read_status,
                (unsigned int)range_mm[0], (unsigned int)range_mm[1],
                (unsigned int)range_mm[2]);
}

void tof_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer == NULL) return;

    vTaskSuspendAll();

    for (uint8_t i = 0; i < 64; i++)
    {
      tof_raw_msg.data.data[i] = Results.distance_mm[i];
        uint8_t status = Results.target_status[i];

        if (status == 5) {
            tof_raw_msg.data.data[i] = Results.distance_mm[i];
        } else {
            tof_raw_msg.data.data[i] = -1; 
        }
    }
    
    xTaskResumeAll(); 

    record_publish(PUB_TOF, &tof_raw_publisher, &tof_raw_msg);
}

void pwm_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      pwm_msg.x = current_pwm_L;
      pwm_msg.y = current_pwm_R;
      pwm_msg.z = 0.0;
      record_publish(PUB_PWM, &pwm_publisher, &pwm_msg);
    }
}

// ==========================================
// MAIN TASK
// ==========================================
void run_microros_app()
{
    uint32_t previous_health_report_ms = 0;
    uint32_t previous_tof_irq_count = 0;
    uint32_t previous_tof_frame_count = 0;

    for (uint8_t i = 0; i < PUB_COUNT; i++) {
        publisher_diag[i].init_ret = RCL_RET_NOT_INIT;
        publisher_diag[i].timer_ret = RCL_RET_NOT_INIT;
        publisher_diag[i].executor_ret = RCL_RET_NOT_INIT;
        publisher_diag[i].last_publish_ret = RCL_RET_NOT_INIT;
    }
    for (uint8_t i = 0; i < SUB_COUNT; i++) {
        subscription_diag[i].init_ret = RCL_RET_NOT_INIT;
        subscription_diag[i].executor_ret = RCL_RET_NOT_INIT;
    }

    // --- 2. Micro-ROS Transport Setup ---
    rmw_uros_set_custom_transport(
      true,
      (void *) &huart1,
      cubemx_transport_open,
      cubemx_transport_close,
      cubemx_transport_write,
      cubemx_transport_read);

    rcl_allocator_t freeRTOS_allocator = rcutils_get_zero_initialized_allocator();
    freeRTOS_allocator.allocate = microros_allocate;
    freeRTOS_allocator.deallocate = microros_deallocate;
    freeRTOS_allocator.reallocate = microros_reallocate;
    freeRTOS_allocator.zero_allocate =  microros_zero_allocate;

    if (!rcutils_set_default_allocator(&freeRTOS_allocator)) {
        printf("Error on default allocators (line %d)\n", __LINE__);
    }

    allocator = rcl_get_default_allocator();

    support_init_ret = rclc_support_init(&support, 0, NULL, &allocator);
    node_init_ret = rclc_node_init_default(&node, "cubemx_node", "", &support);

    // --- 3. WIRING MEMORY ---
    // Publisher Memory

    // WHEEL
    wheel_state_msg.name.data = pub_names;
    wheel_state_msg.name.size = 2;
    wheel_state_msg.name.capacity = 2;

    wheel_state_msg.name.data[0].data = pub_name_L;
    wheel_state_msg.name.data[0].size = strlen(pub_name_L);
    wheel_state_msg.name.data[0].capacity = strlen(pub_name_L) + 1;

    wheel_state_msg.name.data[1].data = pub_name_R;
    wheel_state_msg.name.data[1].size = strlen(pub_name_R);
    wheel_state_msg.name.data[1].capacity = strlen(pub_name_R) + 1;

    wheel_state_msg.position.data = pub_positions;
    wheel_state_msg.position.size = 2;
    wheel_state_msg.position.capacity = 2;

    wheel_state_msg.velocity.data = pub_velocities;
    wheel_state_msg.velocity.size = 2;
    wheel_state_msg.velocity.capacity = 2;

    // TOF
    // Map the internal pointer to our static memory block
    tof_raw_msg.data.data = allocated_data_buffer;
    tof_raw_msg.data.size = 64;
    tof_raw_msg.data.capacity = 64;

    // Configure the multi-array layout parameters to 0 since it's treated as a flat stream
    tof_raw_msg.layout.dim.size = 0;
    tof_raw_msg.layout.dim.capacity = 0;
    tof_raw_msg.layout.dim.data = NULL;
    tof_raw_msg.layout.data_offset = 0;

    // Subscriber Memory
    wheel_cmd_msg.name.data = sub_names;
    wheel_cmd_msg.name.capacity = 2;
    wheel_cmd_msg.name.size = 0;

    wheel_cmd_msg.name.data[0].data = sub_name_string_L;
    wheel_cmd_msg.name.data[0].capacity = 30;
    wheel_cmd_msg.name.data[0].size = 0;

    wheel_cmd_msg.name.data[1].data = sub_name_string_R;
    wheel_cmd_msg.name.data[1].capacity = 30;
    wheel_cmd_msg.name.data[1].size = 0;

    wheel_cmd_msg.velocity.data = sub_velocities;
    wheel_cmd_msg.velocity.capacity = 2;
    wheel_cmd_msg.velocity.size = 0;

    wheel_cmd_msg.position.data = sub_positions;
    wheel_cmd_msg.position.capacity = 2;
    wheel_cmd_msg.position.size = 0;

    wheel_cmd_msg.effort.data = sub_efforts;
    wheel_cmd_msg.effort.capacity = 2;
    wheel_cmd_msg.effort.size = 0;

    // --- 4. Initialize ROS 2 Entities ---
    publisher_diag[PUB_IMU].init_ret = rclc_publisher_init_best_effort(
      &imu_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32), "stm32/imu_msg");

    publisher_diag[PUB_RANGE].init_ret = rclc_publisher_init_best_effort(
      &range_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32), "stm32/range_msg");

    publisher_diag[PUB_TOF].init_ret = rclc_publisher_init_best_effort(
      &tof_raw_publisher, &node, 
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int16MultiArray), "stm32/tof_raw_data"); //Int16MultiArray

    publisher_diag[PUB_DEBUG].init_ret = rclc_publisher_init_best_effort(
      &debug_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String), "stm32/debug_msg");
    publisher_diag[PUB_DEBUG].timer_ret = RCL_RET_OK;
    publisher_diag[PUB_DEBUG].executor_ret = RCL_RET_OK;
    debug_publisher_ready = (publisher_diag[PUB_DEBUG].init_ret == RCL_RET_OK);

    publisher_diag[PUB_PWM].init_ret = rclc_publisher_init_best_effort(
      &pwm_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32), "stm32/pwm_msg");

    publisher_diag[PUB_WHEEL].init_ret = rclc_publisher_init_best_effort(
      &wheel_state_publisher, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
      "stm32/wheel_states");

    subscription_diag[SUB_WHEEL_CMD].init_ret = rclc_subscription_init_default(
      &wheel_cmd_subscriber, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
      "stm32/wheel_commands");

    subscription_diag[SUB_SERVO_TILT].init_ret = rclc_subscription_init_default(
      &servo_tilt_subscriber, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
      "stm32/servo_tilt");

    subscription_diag[SUB_SERVO_PAN].init_ret = rclc_subscription_init_default(
      &servo_pan_subscriber, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
      "stm32/servo_pan");

    publisher_diag[PUB_WHEEL].timer_ret = rclc_timer_init_default2(
        &wheel_timer, &support, RCL_MS_TO_NS(20), wheel_timer_callback, true); // 50Hz
    publisher_diag[PUB_IMU].timer_ret = rclc_timer_init_default2(
        &imu_timer, &support, RCL_MS_TO_NS(20), imu_timer_callback, true); // 50Hz
    publisher_diag[PUB_RANGE].timer_ret = rclc_timer_init_default2(
        &range_timer, &support, RCL_MS_TO_NS(50), range_timer_callback, true); // 20Hz
    publisher_diag[PUB_TOF].timer_ret = rclc_timer_init_default2(
        &tof_timer, &support, RCL_MS_TO_NS(100), tof_timer_callback, true); // 10Hz
    publisher_diag[PUB_PWM].timer_ret = rclc_timer_init_default2(
        &pwm_timer, &support, RCL_MS_TO_NS(50), pwm_timer_callback, true); // 20Hz

    executor_init_ret = rclc_executor_init(&executor, &support.context, 10, &allocator);
    if (executor_init_ret == RCL_RET_OK) {
        if (subscription_diag[SUB_WHEEL_CMD].init_ret == RCL_RET_OK)
            subscription_diag[SUB_WHEEL_CMD].executor_ret = rclc_executor_add_subscription(
                &executor, &wheel_cmd_subscriber, &wheel_cmd_msg, &wheel_cmd_callback, ON_NEW_DATA);
        if (subscription_diag[SUB_SERVO_TILT].init_ret == RCL_RET_OK)
            subscription_diag[SUB_SERVO_TILT].executor_ret = rclc_executor_add_subscription(
                &executor, &servo_tilt_subscriber, &servo_tilt_msg, &servo_tilt_callback, ON_NEW_DATA);
        if (subscription_diag[SUB_SERVO_PAN].init_ret == RCL_RET_OK)
            subscription_diag[SUB_SERVO_PAN].executor_ret = rclc_executor_add_subscription(
                &executor, &servo_pan_subscriber, &servo_pan_msg, &servo_pan_callback, ON_NEW_DATA);

        if (publisher_diag[PUB_WHEEL].init_ret == RCL_RET_OK &&
            publisher_diag[PUB_WHEEL].timer_ret == RCL_RET_OK)
            publisher_diag[PUB_WHEEL].executor_ret = rclc_executor_add_timer(&executor, &wheel_timer);
        if (publisher_diag[PUB_IMU].init_ret == RCL_RET_OK &&
            publisher_diag[PUB_IMU].timer_ret == RCL_RET_OK)
            publisher_diag[PUB_IMU].executor_ret = rclc_executor_add_timer(&executor, &imu_timer);
        if (publisher_diag[PUB_RANGE].init_ret == RCL_RET_OK &&
            publisher_diag[PUB_RANGE].timer_ret == RCL_RET_OK)
            publisher_diag[PUB_RANGE].executor_ret = rclc_executor_add_timer(&executor, &range_timer);
        if (publisher_diag[PUB_TOF].init_ret == RCL_RET_OK &&
            publisher_diag[PUB_TOF].timer_ret == RCL_RET_OK)
            publisher_diag[PUB_TOF].executor_ret = rclc_executor_add_timer(&executor, &tof_timer);
        if (publisher_diag[PUB_PWM].init_ret == RCL_RET_OK &&
            publisher_diag[PUB_PWM].timer_ret == RCL_RET_OK)
            publisher_diag[PUB_PWM].executor_ret = rclc_executor_add_timer(&executor, &pwm_timer);
    }

    for(;;)
    {
        last_executor_spin_ret = rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
        if (last_executor_spin_ret != RCL_RET_OK &&
            last_executor_spin_ret != RCL_RET_TIMEOUT) {
            executor_spin_error_count++;
        }

        uint32_t now = HAL_GetTick();
        uint32_t elapsed = now - previous_health_report_ms;
        if (elapsed >= 1000U) {
            uint32_t current_irq_count = tof_irq_count;
            uint32_t current_frame_count = tof_frame_count;
            uint32_t irq_rate =
                ((current_irq_count - previous_tof_irq_count) * 1000U) / elapsed;
            uint32_t frame_rate =
                ((current_frame_count - previous_tof_frame_count) * 1000U) / elapsed;
            uint32_t irq_age = (tof_last_irq_ms == 0U) ? UINT32_MAX : now - tof_last_irq_ms;
            uint32_t frame_age = (tof_last_frame_ms == 0U) ? UINT32_MAX : now - tof_last_frame_ms;

            publish_health_report(now, elapsed, irq_rate, frame_rate,
                                  irq_age, frame_age);
            previous_tof_irq_count = current_irq_count;
            previous_tof_frame_count = current_frame_count;
            previous_health_report_ms = now;
        }

        osDelay(1);  
    }
}
