#include "main.h"
#include "cmsis_os.h"
#include "dma.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

#include "microros_app.h"
#include "microros_diagnostics.h"
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
extern volatile uint8_t tof_alive_status;
extern volatile uint8_t tof_data_ready;
extern volatile uint32_t tof_irq_count;
extern volatile uint32_t tof_last_irq_ms;
extern volatile uint32_t tof_irq_period_ms;
extern volatile uint32_t tof_frame_count;
extern volatile uint32_t tof_last_frame_ms;
extern volatile uint8_t tof_last_read_status;

#define PI 3.14159265358979323846f
#define WHEEL_RADIUS 0.0325
#define WHEEL_SEPERATION 0.396
#define AGENT_PING_TIMEOUT_MS 100
#define AGENT_PING_ATTEMPTS 3
#define AGENT_RETRY_DELAY_MS 500
#define AGENT_RUNTIME_PING_PERIOD_MS 2000
#define AGENT_RUNTIME_FAILURE_LIMIT 1U
#define ENTITY_CREATION_TIMEOUT_MS 2000
#define ENTITY_DESTROY_TIMEOUT_MS 100
#define IMU_STALE_TIMEOUT_MS 200U
#define RANGE_STALE_TIMEOUT_MS 1000U
#define TOF_STALE_TIMEOUT_MS  500U

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

rcl_publisher_t imu_accel_publisher;
geometry_msgs__msg__Point32 imu_accel_msg;
rcl_publisher_t imu_gyro_publisher;
geometry_msgs__msg__Point32 imu_gyro_msg;

rcl_publisher_t range_publisher;
geometry_msgs__msg__Point32 range_msg;

rcl_publisher_t wheel_state_publisher;
sensor_msgs__msg__JointState wheel_state_msg;

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

static MicroRosDiagnostics diagnostics;
static uint32_t entity_creation_attempts = 0;
static char last_init_failure_stage[32] = "none";
static char last_init_failure_error[128] = "none";
static long last_init_failure_rc = 0;
static rcl_ret_t imu_accel_init_ret = RCL_RET_NOT_INIT;
static rcl_ret_t imu_gyro_init_ret = RCL_RET_NOT_INIT;

// ==========================================
// CALLBACKS & FUNCTIONS
// ==========================================

static bool timestamp_is_fresh(uint32_t now, uint32_t timestamp,
                               uint32_t max_age_ms)
{
    if (timestamp == 0U) {
        return false;
    }

    uint32_t elapsed = now - timestamp;
    if (elapsed > (UINT32_MAX / 2U)) {
        elapsed = 0U;
    }
    return elapsed <= max_age_ms;
}

void servo_tilt_callback(const void * msgin)
{
    const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
    MicroRosDiagnostics_RecordSubscription(
        &diagnostics, SUB_SERVO_TILT, HAL_GetTick());
    Servo_SetTiltTarget(msg->data);
}

void servo_pan_callback(const void * msgin)
{
    const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
    MicroRosDiagnostics_RecordSubscription(
        &diagnostics, SUB_SERVO_PAN, HAL_GetTick());
    Servo_SetPanTarget(msg->data);
}

void wheel_cmd_callback(const void * msgin)
{
    const sensor_msgs__msg__JointState * msg = (const sensor_msgs__msg__JointState *)msgin;

    MicroRosDiagnostics_RecordSubscription(
        &diagnostics, SUB_WHEEL_CMD, HAL_GetTick());

    if (msg->velocity.size >= 2)
    {
        float target_rad_s_L = (float)msg->velocity.data[0];
        float target_rad_s_R = (float)msg->velocity.data[1];

        if (isfinite(target_rad_s_L) && isfinite(target_rad_s_R)) {
            Motor_SetWheelTargets(
                target_rad_s_L * (60.0f / (2.0f * PI)),
                target_rad_s_R * (60.0f / (2.0f * PI)));
        }
    }
}

void wheel_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      wheel_state_msg.position.data[0] = (position_L == 0.0f) ? 1e-6 : (double)position_L;
      wheel_state_msg.position.data[1] = (position_R == 0.0f) ? 1e-6 : (double)position_R;
      wheel_state_msg.velocity.data[0] = (double)(motor_rpm_L * 2.0f * PI / 60.0f);
      wheel_state_msg.velocity.data[1] = (double)(motor_rpm_R * 2.0f * PI / 60.0f);

      MicroRosDiagnostics_Publish(
          &diagnostics, PUB_WHEEL, &wheel_state_publisher, &wheel_state_msg);
    }
}

void imu_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      uint32_t now = HAL_GetTick();
      bool fresh = mpu_init_status && timestamp_is_fresh(
          now, imu_last_read_ms, IMU_STALE_TIMEOUT_MS);
      imu_accel_msg.x = fresh ? Ax : NAN;
      imu_accel_msg.y = fresh ? Ay : NAN;
      imu_accel_msg.z = fresh ? Az : NAN;
      imu_gyro_msg.x = fresh ? Gx : NAN;
      imu_gyro_msg.y = fresh ? Gy : NAN;
      imu_gyro_msg.z = fresh ? Gz : NAN;
      rcl_ret_t publish_result = MicroRosDiagnostics_PublishPair(
          &diagnostics, PUB_IMU,
          &imu_accel_publisher, &imu_accel_msg,
          &imu_gyro_publisher, &imu_gyro_msg);
      if (fresh && publish_result == RCL_RET_OK) {
          MicroRosDiagnostics_RecordImuSample(
              &diagnostics,
              imu_accel_msg.x, imu_accel_msg.y, imu_accel_msg.z,
              imu_gyro_msg.x, imu_gyro_msg.y, imu_gyro_msg.z);
      }
    }
}

void range_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      uint32_t now = HAL_GetTick();
      range_msg.x = Range_IsFresh(0, now, RANGE_STALE_TIMEOUT_MS) ?
          range_mm[0] / 1000.0f : NAN;
      range_msg.y = Range_IsFresh(1, now, RANGE_STALE_TIMEOUT_MS) ?
          range_mm[1] / 1000.0f : NAN;
      range_msg.z = Range_IsFresh(2, now, RANGE_STALE_TIMEOUT_MS) ?
          range_mm[2] / 1000.0f : NAN;
      MicroRosDiagnostics_Publish(
          &diagnostics, PUB_RANGE, &range_publisher, &range_msg);
    }
}

void tof_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer == NULL) return;

    uint32_t now = HAL_GetTick();
    bool fresh = tof_alive_status && timestamp_is_fresh(
        now, tof_last_frame_ms, TOF_STALE_TIMEOUT_MS);

    vTaskSuspendAll();

    for (uint8_t i = 0; i < 64; i++)
    {
        uint8_t status = Results.target_status[i];

        if (!fresh) {
            /* Keep stale frames distinct from sensor target-status failures. */
            tof_raw_msg.data.data[i] = INT16_MIN;
        } else if (status == 5U || status == 9U) {
            tof_raw_msg.data.data[i] = Results.distance_mm[i];
        } else {
            /* Preserve status 0..255 as -1..-256 without another topic. */
            tof_raw_msg.data.data[i] = -(int16_t)(status + 1U);
        }
    }
    
    xTaskResumeAll(); 

    MicroRosDiagnostics_Publish(
        &diagnostics, PUB_TOF, &tof_raw_publisher, &tof_raw_msg);
}

static void wire_message_memory(void)
{
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

    tof_raw_msg.data.data = allocated_data_buffer;
    tof_raw_msg.data.size = 64;
    tof_raw_msg.data.capacity = 64;
    tof_raw_msg.layout.dim.size = 0;
    tof_raw_msg.layout.dim.capacity = 0;
    tof_raw_msg.layout.dim.data = NULL;
    tof_raw_msg.layout.data_offset = 0;

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
}

static void reset_entity_state(void)
{
    support = (rclc_support_t){0};
    node = rcl_get_zero_initialized_node();
    executor = rclc_executor_get_zero_initialized_executor();

    debug_publisher = rcl_get_zero_initialized_publisher();
    imu_accel_publisher = rcl_get_zero_initialized_publisher();
    imu_gyro_publisher = rcl_get_zero_initialized_publisher();
    imu_accel_init_ret = RCL_RET_NOT_INIT;
    imu_gyro_init_ret = RCL_RET_NOT_INIT;
    range_publisher = rcl_get_zero_initialized_publisher();
    wheel_state_publisher = rcl_get_zero_initialized_publisher();
    tof_raw_publisher = rcl_get_zero_initialized_publisher();

    wheel_cmd_subscriber = rcl_get_zero_initialized_subscription();
    servo_tilt_subscriber = rcl_get_zero_initialized_subscription();
    servo_pan_subscriber = rcl_get_zero_initialized_subscription();

    wheel_timer = rcl_get_zero_initialized_timer();
    imu_timer = rcl_get_zero_initialized_timer();
    range_timer = rcl_get_zero_initialized_timer();
    tof_timer = rcl_get_zero_initialized_timer();

    MicroRosDiagnostics_Reset(&diagnostics);
}

static void capture_init_failure(const char *stage, rcl_ret_t ret)
{
    const char *error_text = rcl_get_error_string().str;

    strncpy(last_init_failure_stage, stage, sizeof(last_init_failure_stage) - 1U);
    last_init_failure_stage[sizeof(last_init_failure_stage) - 1U] = '\0';
    last_init_failure_rc = (long)ret;

    if (error_text != NULL && error_text[0] != '\0') {
        strncpy(last_init_failure_error, error_text,
                sizeof(last_init_failure_error) - 1U);
        last_init_failure_error[sizeof(last_init_failure_error) - 1U] = '\0';
    } else {
        strcpy(last_init_failure_error, "not_set");
    }
    rcl_reset_error();
}

#define INIT_OR_FAIL(destination, expression, stage_name) \
    do { \
        (destination) = (expression); \
        if ((destination) != RCL_RET_OK) { \
            capture_init_failure((stage_name), (destination)); \
            return false; \
        } \
    } while (0)

static bool create_microros_entities(void)
{
    rmw_context_t *rmw_context;

    entity_creation_attempts++;
    reset_entity_state();
    rcl_reset_error();

    INIT_OR_FAIL(diagnostics.support_init_ret,
                 rclc_support_init(&support, 0, NULL, &allocator),
                 "support");

    rmw_context = rcl_context_get_rmw_context(&support.context);
    if (rmw_context != NULL) {
        (void)rmw_uros_set_context_entity_creation_session_timeout(
            rmw_context, ENTITY_CREATION_TIMEOUT_MS);
    }

    INIT_OR_FAIL(diagnostics.node_init_ret,
                 rclc_node_init_default(&node, "cubemx_node", "", &support),
                 "node");

    INIT_OR_FAIL(imu_accel_init_ret,
                 rclc_publisher_init_best_effort(
                     &imu_accel_publisher, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32),
                     "stm32/imu_accel"),
                 "pub_imu_accel");
    INIT_OR_FAIL(imu_gyro_init_ret,
                 rclc_publisher_init_best_effort(
                     &imu_gyro_publisher, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32),
                     "stm32/imu_gyro"),
                 "pub_imu_gyro");
    diagnostics.publishers[PUB_IMU].init_ret = RCL_RET_OK;
    INIT_OR_FAIL(diagnostics.publishers[PUB_RANGE].init_ret,
                 rclc_publisher_init_best_effort(
                     &range_publisher, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32),
                     "stm32/range_msg"),
                 "pub_range");
    INIT_OR_FAIL(diagnostics.publishers[PUB_TOF].init_ret,
                 rclc_publisher_init_best_effort(
                     &tof_raw_publisher, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int16MultiArray),
                     "stm32/tof_raw_data"),
                 "pub_tof");
    INIT_OR_FAIL(diagnostics.publishers[PUB_DEBUG].init_ret,
                 rclc_publisher_init_best_effort(
                     &debug_publisher, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
                     "stm32/debug_msg"),
                 "pub_debug");
    diagnostics.publishers[PUB_DEBUG].timer_ret = RCL_RET_OK;
    diagnostics.publishers[PUB_DEBUG].executor_ret = RCL_RET_OK;

    INIT_OR_FAIL(diagnostics.publishers[PUB_WHEEL].init_ret,
                 rclc_publisher_init_best_effort(
                     &wheel_state_publisher, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
                     "stm32/wheel_states"),
                 "pub_wheel");

    /* Wheel commands are state, not events. Best effort prevents an old
       reliable command from being replayed after a transport outage. */
    INIT_OR_FAIL(diagnostics.subscriptions[SUB_WHEEL_CMD].init_ret,
                 rclc_subscription_init_best_effort(
                     &wheel_cmd_subscriber, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
                     "stm32/wheel_commands"),
                 "sub_wheel_cmd");
    INIT_OR_FAIL(diagnostics.subscriptions[SUB_SERVO_TILT].init_ret,
                 rclc_subscription_init_default(
                     &servo_tilt_subscriber, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
                     "stm32/servo_tilt"),
                 "sub_servo_tilt");
    INIT_OR_FAIL(diagnostics.subscriptions[SUB_SERVO_PAN].init_ret,
                 rclc_subscription_init_default(
                     &servo_pan_subscriber, &node,
                     ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
                     "stm32/servo_pan"),
                 "sub_servo_pan");

    INIT_OR_FAIL(diagnostics.publishers[PUB_WHEEL].timer_ret,
                 rclc_timer_init_default2(&wheel_timer, &support,
                     RCL_MS_TO_NS(20), wheel_timer_callback, true),
                 "timer_wheel");
    INIT_OR_FAIL(diagnostics.publishers[PUB_IMU].timer_ret,
                 rclc_timer_init_default2(&imu_timer, &support,
                     RCL_MS_TO_NS(20), imu_timer_callback, true),
                 "timer_imu");
    INIT_OR_FAIL(diagnostics.publishers[PUB_RANGE].timer_ret,
                 rclc_timer_init_default2(&range_timer, &support,
                     RCL_MS_TO_NS(50), range_timer_callback, true),
                 "timer_range");
    INIT_OR_FAIL(diagnostics.publishers[PUB_TOF].timer_ret,
                 rclc_timer_init_default2(&tof_timer, &support,
                     RCL_MS_TO_NS(100), tof_timer_callback, true),
                 "timer_tof");
    INIT_OR_FAIL(diagnostics.executor_init_ret,
                 rclc_executor_init(&executor, &support.context, 7, &allocator),
                 "executor");

    INIT_OR_FAIL(diagnostics.subscriptions[SUB_WHEEL_CMD].executor_ret,
                 rclc_executor_add_subscription(
                     &executor, &wheel_cmd_subscriber, &wheel_cmd_msg,
                     &wheel_cmd_callback, ON_NEW_DATA),
                 "exec_wheel_cmd");
    INIT_OR_FAIL(diagnostics.subscriptions[SUB_SERVO_TILT].executor_ret,
                 rclc_executor_add_subscription(
                     &executor, &servo_tilt_subscriber, &servo_tilt_msg,
                     &servo_tilt_callback, ON_NEW_DATA),
                 "exec_servo_tilt");
    INIT_OR_FAIL(diagnostics.subscriptions[SUB_SERVO_PAN].executor_ret,
                 rclc_executor_add_subscription(
                     &executor, &servo_pan_subscriber, &servo_pan_msg,
                     &servo_pan_callback, ON_NEW_DATA),
                 "exec_servo_pan");
    INIT_OR_FAIL(diagnostics.publishers[PUB_WHEEL].executor_ret,
                 rclc_executor_add_timer(&executor, &wheel_timer),
                 "exec_wheel");
    INIT_OR_FAIL(diagnostics.publishers[PUB_IMU].executor_ret,
                 rclc_executor_add_timer(&executor, &imu_timer),
                 "exec_imu");
    INIT_OR_FAIL(diagnostics.publishers[PUB_RANGE].executor_ret,
                 rclc_executor_add_timer(&executor, &range_timer),
                 "exec_range");
    INIT_OR_FAIL(diagnostics.publishers[PUB_TOF].executor_ret,
                 rclc_executor_add_timer(&executor, &tof_timer),
                 "exec_tof");
    MicroRosDiagnostics_SetDebugReady(&diagnostics, true);
    return true;
}

#undef INIT_OR_FAIL

static void destroy_microros_entities(bool agent_available)
{
#define CLEANUP_RCL(expression) \
    do { \
        rcl_ret_t cleanup_ret = (expression); \
        if (cleanup_ret != RCL_RET_OK) { \
            rcl_reset_error(); \
        } \
    } while (0)

    MicroRosDiagnostics_SetDebugReady(&diagnostics, false);

    if (diagnostics.support_init_ret == RCL_RET_OK) {
        rmw_context_t *rmw_context = rcl_context_get_rmw_context(&support.context);
        if (rmw_context != NULL) {
            (void)rmw_uros_set_context_entity_destroy_session_timeout(
                rmw_context, agent_available ? ENTITY_DESTROY_TIMEOUT_MS : 0);
        }
    }

    if (diagnostics.executor_init_ret == RCL_RET_OK) {
        CLEANUP_RCL(rclc_executor_fini(&executor));
    }

    if (diagnostics.publishers[PUB_TOF].timer_ret == RCL_RET_OK) CLEANUP_RCL(rcl_timer_fini(&tof_timer));
    if (diagnostics.publishers[PUB_RANGE].timer_ret == RCL_RET_OK) CLEANUP_RCL(rcl_timer_fini(&range_timer));
    if (diagnostics.publishers[PUB_IMU].timer_ret == RCL_RET_OK) CLEANUP_RCL(rcl_timer_fini(&imu_timer));
    if (diagnostics.publishers[PUB_WHEEL].timer_ret == RCL_RET_OK) CLEANUP_RCL(rcl_timer_fini(&wheel_timer));

    if (diagnostics.node_init_ret == RCL_RET_OK) {
        if (diagnostics.subscriptions[SUB_SERVO_PAN].init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_subscription_fini(&servo_pan_subscriber, &node));
        if (diagnostics.subscriptions[SUB_SERVO_TILT].init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_subscription_fini(&servo_tilt_subscriber, &node));
        if (diagnostics.subscriptions[SUB_WHEEL_CMD].init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_subscription_fini(&wheel_cmd_subscriber, &node));

        if (diagnostics.publishers[PUB_WHEEL].init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_publisher_fini(&wheel_state_publisher, &node));
        if (diagnostics.publishers[PUB_DEBUG].init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_publisher_fini(&debug_publisher, &node));
        if (diagnostics.publishers[PUB_TOF].init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_publisher_fini(&tof_raw_publisher, &node));
        if (diagnostics.publishers[PUB_RANGE].init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_publisher_fini(&range_publisher, &node));
        if (imu_gyro_init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_publisher_fini(&imu_gyro_publisher, &node));
        if (imu_accel_init_ret == RCL_RET_OK)
            CLEANUP_RCL(rcl_publisher_fini(&imu_accel_publisher, &node));

        CLEANUP_RCL(rcl_node_fini(&node));
    }

    if (diagnostics.support_init_ret == RCL_RET_OK) {
        CLEANUP_RCL(rclc_support_fini(&support));
    }
    rcl_reset_error();

#undef CLEANUP_RCL
}

// ==========================================
// MAIN TASK
// ==========================================
void run_microros_app()
{
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
    wire_message_memory();

    for (;;) {
        uint32_t failed_runtime_pings = 0U;

        while (rmw_uros_ping_agent(AGENT_PING_TIMEOUT_MS,
                                   AGENT_PING_ATTEMPTS) != RMW_RET_OK) {
            osDelay(AGENT_RETRY_DELAY_MS);
        }

        if (!create_microros_entities()) {
            destroy_microros_entities(true);
            osDelay(AGENT_RETRY_DELAY_MS);
            continue;
        }

        uint32_t previous_agent_ping_ms = HAL_GetTick();
        diagnostics.previous_health_report_ms = previous_agent_ping_ms;
        diagnostics.previous_system_report_ms = previous_agent_ping_ms;
        diagnostics.previous_tof_irq_count = tof_irq_count;
        diagnostics.previous_tof_frame_count = tof_frame_count;

        if (entity_creation_attempts > 1U) {
            MicroRosDiagnostics_Log(
                &diagnostics, &debug_publisher,
                "[MICROROS] session_recreated attempts=%lu last_stage=%s rc=%ld error=%s",
                (unsigned long)entity_creation_attempts,
                last_init_failure_stage, last_init_failure_rc,
                last_init_failure_error);
        } else {
            MicroRosDiagnostics_Log(
                &diagnostics, &debug_publisher,
                "[MICROROS] connected attempts=%lu",
                (unsigned long)entity_creation_attempts);
        }
        MicroRosDiagnostics_ReportBoot(&diagnostics, &debug_publisher);

        for (;;) {
            rcl_ret_t spin_result =
                rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
            MicroRosDiagnostics_RecordExecutor(
                &diagnostics, spin_result, HAL_GetTick());

            uint32_t now = HAL_GetTick();
            if ((now - previous_agent_ping_ms) >=
                AGENT_RUNTIME_PING_PERIOD_MS) {
                rmw_ret_t ping_result = rmw_uros_ping_agent(
                    AGENT_PING_TIMEOUT_MS, AGENT_PING_ATTEMPTS);
                MicroRosDiagnostics_RecordAgentPing(
                    &diagnostics, ping_result, now);
                previous_agent_ping_ms = now;

                if (ping_result == RMW_RET_OK) {
                    failed_runtime_pings = 0U;
                } else if (++failed_runtime_pings >=
                           AGENT_RUNTIME_FAILURE_LIMIT) {
                    Motor_SetWheelTargets(0.0f, 0.0f);
                    MicroRosDiagnostics_Log(
                        &diagnostics, &debug_publisher,
                        "[MICROROS] session_lost failed_ping_checks=%lu",
                        (unsigned long)failed_runtime_pings);
                    break;
                }
            }

            MicroRosDiagnostics_ReportIfDue(
                &diagnostics, &debug_publisher, now);
            osDelay(1);
        }

        /* No more executor spins are allowed after declaring the link lost,
           so a queued motion command cannot refresh the motor watchdog. */
        destroy_microros_entities(false);
        osDelay(AGENT_RETRY_DELAY_MS);
    }
}
