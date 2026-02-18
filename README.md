# 🧠 NeuraRig

**NeuraRig** is a C++20 library for **Neural Inverse Kinematics (IK)** and procedural character animation.

Unlike traditional Control Rigs that require complex manual math, NeuraRig uses **Deep Learning** to solve character poses. It learns natural movement patterns from data and applies them in real-time using **LibTorch**.

---

## 🚀 Stage 2: Real-Time Data Acquisition & Dynamic Mapping (Active)

The foundation now bridges Game Engines and Neural Training with a **Dynamic Skeleton Protocol**. We no longer rely on hardcoded bone counts; the rig "explains" itself to the AI.

### 📡 Smart Data Streaming
We've implemented a dual-protocol UDP layer that handles both configuration and motion:

* **Dynamic Bone Mapping (0x01):** A handshake protocol where the Engine (UE5) sends the Rig hierarchy (ID + Name) via serialized ANSI strings. NeuraRig builds a `BoneMap` on-the-fly.
* **High-Speed Motion Stream (0x02):** Optimized 12-byte blocks (X, Y, Z floats) per bone. Zero-copy memory mapping directly into PyTorch Tensors.

### 🛠️ Key Architecture Updates
* **NRRigDescription:** Now features an `unordered_map` for O(1) bone lookups and automatic `InputSize` calculation.

### 📽️ Progress Preview: Bone Tracking
*(Console output shows the new Auto-Mapping: `Mapped Bone: 0 -> foot_r, 1 -> foot_l` followed by real-time coordinates)*
<p align="center">
  <img src="GIFProgress.gif" alt="NeuraRig Bone Tracking" width="800">
</p>
*(Console output: Real-time X, Y, Z coordinates being streamed for 161+ bones per frame)*

---

## 🎯 Current Status: Integration Success
1. **Memory Alignment:** 1-to-1 mapping of UDP buffers to LibTorch tensors confirmed.
2. **Optimizer Stability:** MSE reduction confirmed; the "brain" is officially learning.
3. **Dynamic Scaling:** The server now adapts to any number of bones without recompilation.

**Next Step:** Implementation of the **NRSolver** for real-time inference and **Data Buffering** to store long-term motion sequences for complex training.
