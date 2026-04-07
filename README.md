# 🧠 NeuraRig

**NeuraRig** is a high-performance C++20 library for **Neural Inverse Kinematics (IK)** and procedural character animation.

Unlike traditional, mathematically rigid IK solvers, NeuraRig leverages a **Hybrid Neural Approach**. It combines deterministic gait physics with **Deep Learning** to solve complex skeletal hierarchies, ensuring biologically plausible movement with the performance of a specialized **LibTorch (PyTorch C++)** backend.

[![Status](https://img.shields.io/badge/status-active--development-green)](#)
[![C++](https://img.shields.io/badge/language-C%2B%2B20-blue)](https://en.cppreference.com/w/cpp/20)
[![LibTorch](https://img.shields.io/badge/backend-LibTorch-red)](https://pytorch.org/cppdocs/)

---

## 🚀 Breakthrough: Hybrid Neural Stride Logic

NeuraRig has reached a new milestone in character locomotion, merging pure neural inference with Forward Kinematics (FK) constraints.

#### 📊 Training Performance Logs
```text
Frame: 52441
BaseLoss: 0.0168113
TemporalLoss: 0.000620833
AccelLoss: 0.000446737
TotalLoss: 0.0172751
----------------------------------
Frame: 52471
BaseLoss: 0.0101665
TemporalLoss: 0.00218407
AccelLoss: 0.0012738
TotalLoss: 0.0119785
----------------------------------
[Checkpoint] Modelo salvo automaticamente em: ../trained_model.pt (Frame: 52500)
Frame: 52501
BaseLoss: 0.00211306
TemporalLoss: 0.000299212
AccelLoss: 0.000308155
TotalLoss: 0.0023765
----------------------------------
Frame: 52531
BaseLoss: 0.00770069
TemporalLoss: 0.000274027
AccelLoss: 0.000155135
TotalLoss: 0.0079959
----------------------------------
Frame: 52561
BaseLoss: 0.00259855
TemporalLoss: 0.000113437
AccelLoss: 0.000142221
TotalLoss: 0.00272894
```

### ⚙️ The Gait Engine (Schema v3)
NeuraRig uses a programmable logic layer to define movement patterns. This allows the AI to adapt to different skeletal proportions and velocities dynamically.

*   **Temporal Awareness:** Uses `t_cycle` and `time_dilation` to maintain a continuous, jitter-free motion loop.
*   **Anatomy Agnostic:** Bone lengths (`bone_l1` to `bone_l4`) are fed directly into the logic, allowing the model to scale strides for any character size automatically.
*   **Phase Switching:** Computes transitions between **Stance** and **Swing** phases based on a normalized `linear_cycle`.

---

## 🛠️ Architecture & Features

The library processes locomotion parameters and outputs precise IK targets and joint rotations.

### Data Flow (Tensor Mapping)
The system now handles a more complex schema to support full leg articulation:

**Inputs (18 floats):**
- Velocity
- Bone Lengths (L1-L4 for both legs)
- Ground Hit status
- Phase Offsets & Spacing
- Global Cycle Time & Dilation

**Outputs (60 floats / 10 Targets):**
- **IK Targets (Vec3 + Rot):** Foot (R/L), Ball (R/L), Pelvis, Spine.
- **FK Targets (Vec3 + Rot):** Calf (R/L) and Thigh (R/L) rotations generated via the pre-trained FK model.

### 🌟 New Features

*   **Pre-trained FK Model:** A specialized model that generates thigh and knee (calf) rotations based on foot positioning, ensuring natural leg bending without manual IK constraints.
*   **Temporal Smoothing:** Implements **EMA (Exponential Moving Average)** and acceleration loss to eliminate jitter and produce fluid, cinematic motion.
*   **Best Pose Scoring:** The system generates multiple prediction candidates and selects the optimal pose based on a composite score of precision, velocity consistency, and physical constraints.

---

## 🚧 Project Status: Active Research

> [!IMPORTANT]
> This repository is in an intensive research stage.
> - **Current Focus:** Refinement of the **FK-IK Hybrid Loop** and multi-candidate scoring.
> - **Performance:** Optimized for real-time inference in high-fidelity simulation environments.

---

## 📺 Development Progress

![FK Rig Logic 1](FK_Rig01.gif)

![FK Rig Logic 2](FK_Rig02.gif)

*The AI successfully learns to optimize the stride curve, achieving fluid movement and maintaining balance through real-time weight shift predictions and FK-validated joint rotations.*

---

## 🛠️ Key Components

*   **NRMLPModel:** Multi-layer perceptron optimized for high-dimensional spatial regression.
*   **NRTrainee:** Advanced trainer handling backpropagation with FK-loss and temporal smoothing.
*   **NRSolver:** Real-time inference engine for seamless integration.
*   **Protocol 0x03:** High-speed neural prediction feedback loop with candidate scoring.

---

## 🎮 Unreal Engine Integration

To use **NeuraRig** within **Unreal Engine**, you must install the dedicated plugin. This plugin acts as the bridge, handling data preprocessing and communication between the engine and the NeuraRig library.

*   **Plugin Repository:** [Neura-Rig-Unreal](https://github.com/rafaelvaloto/Neura-Rig-Unreal)
*   **Key Functionality:** Real-time data stream, skeletal mapping, and neural inference integration directly into Unreal's Animation Blueprints.

---

## 🛠️ Dependencies & Installation

NeuraRig depends on several libraries to handle JSON parsing, mathematical expressions, and neural network inference.

### 📦 Required Libraries

1.  **nlohmann_json:** Used for configuration and schema parsing.
    -   Clone the repository into `3rdParty/json`:
        ```bash
        git clone https://github.com/nlohmann/json.git 3rdParty/json
        ```
2.  **muparser:** Used for gait logic expression evaluation.
    -   Clone the repository into `3rdParty/muparser`:
        ```bash
        git clone https://github.com/beltoforion/muparser.git 3rdParty/muparser
        ```
3.  **LibTorch (PyTorch C++ Frontend):** The core neural engine.
    -   Download the Windows shared CPU version (v2.11.0): [libtorch-win-shared-with-deps-2.11.0+cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.11.0%2Bcpu.zip)
    -   Extract the contents into `3rdParty/libtorch`.

---

## 🏃 Getting Started

To use the pre-trained model in your builds:
1.  Locate the model `trained_model.pt` in `Tests/Datasets/`.
2.  After compiling the project, move/copy this file to your build directory (e.g., `cmake-build-release/Tests/Datasets/` or where the executable is located) to ensure the tests can load the weights correctly.
