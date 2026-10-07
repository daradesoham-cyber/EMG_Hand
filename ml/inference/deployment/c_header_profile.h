/**
 * @file c_header_profile.h
 * @brief Embedded C deployment definitions for Causal TCN Prosthetic Controller.
 * @details Target: Prospective ESP32-S3 / ARM Cortex-M microcontrollers.
 */

#ifndef CAUSAL_TCN_PROFILE_H
#define CAUSAL_TCN_PROFILE_H

#ifdef __cplusplus
extern "C" {
#endif

/* System & Sampling Constants */
#define EMG_SAMPLING_RATE_HZ        2000
#define EMG_NUM_CHANNELS            16
#define EMG_WINDOW_SAMPLES          256     /* 128.0 ms history */
#define EMG_STRIDE_SAMPLES          64      /* 32.0 ms step */
#define EMG_INFERENCE_FREQ_HZ       31.25f

/* Actuator Output Constants */
#define HASEL_NUM_CHANNELS          6
#define HASEL_H1_INDEX              0
#define HASEL_H2_MIDDLE             1
#define HASEL_H3_RING               2
#define HASEL_H4_LITTLE             3
#define HASEL_H5_THUMB_FLEXION      4
#define HASEL_H6_THUMB_OPPOSITION   5

/* Post-Processing Exponential Moving Average (EMA) */
#define HASEL_EMA_ALPHA             0.20f   /* Time constant tau = 143.4 ms */

/* Memory Footprint Accounting */
#define TCN_TOTAL_PARAMETERS        332966
#define TCN_WEIGHT_SIZE_INT8_BYTES  332966
#define TCN_EST_SRAM_TOTAL_BYTES    662700  /* ~647 kB */

/* Signal Conditioning Bounds */
#define FILTER_BP_LOW_HZ            20.0f
#define FILTER_BP_HIGH_HZ           450.0f
#define FILTER_NOTCH_HZ             50.0f
#define FILTER_NOTCH_Q              30.0f

#ifdef __cplusplus
}
#endif

#endif /* CAUSAL_TCN_PROFILE_H */
