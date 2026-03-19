# 🧠 NeuraRig

**NeuraRig** is a high-performance C++20 library for **Neural Inverse Kinematics (IK)** and procedural character animation.

Unlike traditional, mathematically rigid IK solvers, NeuraRig leverages a **Hybrid Neural Approach**. It combines deterministic gait physics with **Deep Learning** to solve complex skeletal hierarchies, ensuring biologically plausible movement with the performance of a specialized **LibTorch (PyTorch C++)** backend.

[![Status](https://img.shields.io/badge/status-research--prototype-orange)](#)
[![C++](https://img.shields.io/badge/language-C%2B%2B20-blue)](https://en.cppreference.com/w/cpp/20)
[![LibTorch](https://img.shields.io/badge/backend-LibTorch-red)](https://pytorch.org/cppdocs/)

---

## 🚀 Breakthrough: Hybrid Neural Stride Logic

[🎥 Click and watch the example video on YouTube.](https://www.youtube.com/watch?v=1hWfmAZ5whk)

The project has evolved to **Stage 3**, introducing a logic-driven schema that merges pure neural inference with procedural animation rules. The system now "understands" the phases of a gait cycle (Stance vs. Swing) in real-time across multiple skeletal components.

### 📉 Precision & Convergence
The neural optimizer demonstrates extreme stability during active training:
* **Rapid Learning:** Convergence from an initial Loss of **~685.12** to sub-millimeter precision in under **800 frames**.
* **Deep Stability:** Achieving a precision threshold as low as **2.85e-07**, reaching near-perfect mathematical alignment.
* **Surgical Accuracy:** The real-time delta between Unreal Engine's ground truth and AI prediction is consistently below **0.0001 units**.

### ⚙️ The Gait Engine (Schema v3)
NeuraRig uses a programmable logic layer to define movement patterns. This allows the AI to adapt to different skeletal proportions and velocities dynamically.

* **Temporal Awareness:** Uses `t_cycle` and `t_accumulated` to maintain a continuous, jitter-free motion loop.
* **Anatomy Agnostic:** By inputting bone lengths (`bone_l1`, `bone_l2`) directly into the logic, the model automatically scales the stride for any character size.
* **Phase Switching:** The engine computes the transition between **Stance** (fixed ground contact) and **Swing** (sinusoidal arc) based on a normalized `linear_cycle`.

---

## 🛠️ Architecture & Schema

The library processes high-level locomotion parameters and outputs precise IK targets in **Local Space (LS)**.

### Data Flow (Tensor Mapping)
```text
Inputs (11 floats):
┌──────────┬─────────────┬─────────────┬──────────┬─────────────┬───────────┬─────────────┐
│ Velocity │ Bone L1 R/L │ Bone L2 R/L │ Offsets  │ Spacing R/L │ Cycle T   │ Accumulated │
│ [0]      │ [1, 5]      │ [2, 6]      │ [3, 7]   │ [4, 8]      │ [9]       │ [10]        │
└──────────┴─────────────┴─────────────┴──────────┴─────────────┴───────────┴─────────────┘

Outputs (30 floats / Vec3 + Rot):
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│     Foot R/L (6+6)       │     Ball R/L (6+6)       │     Pelvis (6)           │
│     [0 - 11]             │     [12 - 23]            │     [24 - 29]            │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

### ⚙️ Gait Engine & Bindings (Schema v3)
NeuraRig uses a programmable logic layer to define complex movement patterns, allowing the AI to dynamically adapt to different skeletal proportions and velocities.

*   **Temporal Awareness:** Uses `t_cycle` and `t_accumulated` to maintain a continuous, jitter-free motion loop.
*   **Anatomy Agnostic:** By providing bone lengths (`bone_l1`, `bone_l2`) directly in the logic, the model automatically scales the stride for any character size.
*   **Multi-Target Bindings:** The system now automatically maps procedural animation curves to multiple targets:
    *   **Foot IK:** Control of position and pitch of the feet.
    *   **Ball IK:** Joint articulation (metatarsals) for natural strides.
    *   **Pelvis IK:** Sway, vertical oscillation (bob), and rotation (yaw/roll) movements of the pelvis based on the walking cycle.
*   **Phase Switching:** The engine computes the transition between **Stance** (fixed ground contact) and **Swing** (sinusoidal arc) based on a normalized `linear_cycle`.

## 🚧 Project Status: Prototyping

> [!IMPORTANT]
> This repository is in its **earliest research stage**.
> - **Stability:** Highly unstable; breaking changes occur daily.
> - **Usability:** No MVP is currently available for public functional use.
> - **Current Focus:** Transitioning to **Time-Series Prediction** for autonomous gait generation and refined foot-planting.

---

## 📺 Development Progress

![Neural IK Demo](atGif.gif)

*The AI successfully learns to optimize the stride curve, achieving fluid movement and maintaining balance through real-time weight shift predictions.*

---

## 🛠️ Key Components

* **NRMLPModel:** Multi-layer perceptron optimized for spatial regression.
* **NRTrainee:** Handles backpropagation and weights persistence (`rig_model.pt`).
* **NRSolver:** Real-time inference engine for game engine integration.
* **Protocol 0x03:** High-speed neural prediction feedback loop.
