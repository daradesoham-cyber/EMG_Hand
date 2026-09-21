# BiDexHand Hardware Mapping & EMG Electrode Placement

This document details the hardware mapping for the 16-servo BiDexHand and the physical sEMG electrode placement protocol for the V1 research prototype.

---

## 1. 16-Servo Joint Mapping Table

Target angles for each gesture in degrees ($\pm$ indicates flexion/extension or abduction/adduction):

| Index | Joint Name | Joint Description | REST ($^\circ$) | FIST ($^\circ$) | OPEN ($^\circ$) | PINCH ($^\circ$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | `tca_joint` | Thumb CMC Abduction | 0.0 | 0.0 | 10.0 | -15.0 |
| **1** | `ima_joint` | Index MCP Abduction | 0.0 | 0.0 | -10.0 | 0.0 |
| **2** | `mma_joint` | Middle MCP Abduction | 0.0 | 0.0 | 0.0 | 0.0 |
| **3** | `rma_joint` | Ring MCP Abduction | 0.0 | 0.0 | 5.0 | 0.0 |
| **4** | `pma_joint` | Pinky MCP Abduction | 0.0 | 0.0 | 15.0 | 0.0 |
| **5** | `tma_joint` | Thumb MCP Abduction | 0.0 | 0.0 | -15.0 | 35.0 |
| **6** | `imf_joint` | Index MCP Flexion | 0.0 | 50.0 | -15.0 | 45.0 |
| **7** | `mmf_joint` | Middle MCP Flexion | 0.0 | 50.0 | -15.0 | 0.0 |
| **8** | `rmf_joint` | Ring MCP Flexion | 0.0 | 50.0 | -15.0 | 0.0 |
| **9** | `pmf_joint` | Pinky MCP Flexion | 0.0 | 50.0 | -15.0 | 0.0 |
| **10**| `tmf_joint` | Thumb MCP Flexion | 0.0 | -30.0 | -15.0 | -40.0 |
| **11**| `ipf_joint` | Index PIP Flexion | 0.0 | 80.0 | -10.0 | 60.0 |
| **12**| `mpf_joint` | Middle PIP Flexion | 0.0 | 80.0 | -10.0 | 0.0 |
| **13**| `rpf_joint` | Ring PIP Flexion | 0.0 | 80.0 | -10.0 | 0.0 |
| **14**| `ppf_joint` | Pinky PIP Flexion | 0.0 | 80.0 | -10.0 | 0.0 |
| **15**| `tcf_joint` | Thumb CMC Flexion | 0.0 | -45.0 | 0.0 | -35.0 |

---

## 2. Surface EMG (sEMG) Electrode Placement

The V1 model utilizes 4 differential electrode channels (1-based indices `[12, 11, 7, 2]`):

```
       [ Anterior / Palmar View ]                 [ Posterior / Dorsal View ]
              (Flexor Side)                              (Extensor Side)
                 Elbow                                       Elbow
                   |                                           |
         +---------+---------+                       +---------+---------+
         |     [Channel 12]  |                       |     [Channel 7]   |
         | (Flexor Digitorum |                       | (Extensor Digitorum|
         |   Superficialis)  |                       |     Communis)     |
         |                   |                       |                   |
         |     [Channel 11]  |                       |     [Channel 2]   |
         | (Flexor Carpi     |                       | (Extensor Carpi   |
         |   Radialis)       |                       |     Ulnaris)      |
         +---------+---------+                       +---------+---------+
                   |                                           |
                 Wrist                                       Wrist
```

### Channel Details:
- **Channel 12**: Placed over *Flexor Digitorum Superficialis* (FDS) — activated during finger flexion (FIST, PINCH).
- **Channel 11**: Placed over *Flexor Carpi Radialis* (FCR) — activated during wrist flexion and thumb opposition.
- **Channel 7**: Placed over *Extensor Digitorum Communis* (EDC) — activated during finger extension and splaying (OPEN).
- **Channel 2**: Placed over *Extensor Carpi Ulnaris* (ECU) — activated during hand stabilization and wrist extension.
- **Reference / Ground**: Placed over an electrically neutral bony prominence (e.g., the olecranon process of the elbow or the styloid process of the ulna).

---

## 3. Physical Hardware Integration Considerations

1. **Sampling Rate**: Analog-to-Digital Converter (ADC) must sample at $\ge 1000\text{ Hz}$ per channel with $\ge 12\text{-bit}$ resolution.
2. **Skin Preparation**: Clean skin with 70% isopropyl alcohol and apply conductive gel or use silver-silver chloride (Ag/AgCl) electrodes.
3. **RMS Calibration**: The software inactivity threshold (`rms_threshold = 0.01`) must be tuned to the resting baseline noise of the physical frontend amplifier.
