# 🧠 NeuraRig

**NeuraRig** is a C++20 library for **Neural Inverse Kinematics (IK)** and procedural character animation.

Unlike traditional rigs that rely on heavy manual mathematics, NeuraRig uses **Deep Learning** to solve character poses. It learns natural movement patterns from raw data and applies them in real-time using a specialized **LibTorch** backend.

---

## 🚀 Breakthrough: Live Neural Solving & High-Precision Convergence

The project has successfully reached **Stage 3**, moving beyond simple data streaming into **Active Neural Inference**. The AI can now learn a character's anatomy in real-time and predict complex bone dependencies (such as Knee/Calf positions) based solely on reference inputs (Feet/Foot positions).

### 📉 Real-Time Learning & Convergence
The system demonstrates massive error reduction during live sessions, proving the stability of the neural optimizer:
* **Initial State:** High-error variance with a starting Loss of ~685.12.
* **Converged State:** Achieving a precision threshold below 0.001 within ~800 frames.
* **Stability:** Reaching deep convergence with a Loss as low as **2.85e-07**, indicating near-perfect mathematical alignment.

### 🤖 Intelligent NRSolver
Once the model reaches the precision threshold, NeuraRig automatically initializes the **NRSolver**. This allows the AI to take control of the rig, predicting the position of actuator bones with extreme accuracy:
* **Real-Time Accuracy:** The difference between Unreal Engine's "Real" position and the AI's "Predicted" position is often less than **0.0001 units**.
* **Reference Bone Driven:** By simply passing the coordinates of the feet, the solver instantly returns the anatomically correct position of the calves.
* **Parent-Space Optimization:** By processing data in local parent-bone space, the model remains stable regardless of the character's global position in the world.

### 📡 Updated Protocol & Workflow
The library now manages a full-circle communication loop between the Game Engine and the Neural Server:
* **Dynamic Handshake (0x01):** Auto-configures the Rig hierarchy and mapping.
* **Active Trainee (0x02):** Feeds real-time motion data into the optimizer.
* **Neural Prediction (0x03):** Sends the AI-generated bone positions back to the Engine for procedural bone modification.

---

## 🎯 Current Benchmarks: Real vs. Predicted
During active solving, the AI-generated coordinates follow the skeletal constraints with surgical precision:

* **Training Stability**: Smooth MSE reduction from ~685.12 to sub-millimeter precision.
* **Reconstruction Fidelity**:
    * **Real Input (X):** 45.8193
    * **AI Prediction (X):** 45.8194
* **Result**: The AI-generated motion is mathematically and visually indistinguishable from the source data.

---

```aiignore
Input (63 floats):
┌─────────┬──────────┬─────────────────────┬─────────────────────┬──────────────────┐
│ pelvis  │ pelvis   │  thigh/calf/foot_r  │  thigh/calf/foot_l  │ targets + normals│
│ pos(3)  │ quat(4)  │  pos+quat ×3 (21)   │  pos+quat ×3 (21)   │ + hits (14)      │
│ [0..2]  │ [3..6]   │  [7..27]            │  [28..48]           │ [49..62]         │
└─────────┴──────────┴─────────────────────┴─────────────────────┴──────────────────┘

Output (24 floats):
┌──────────────────────────┬──────────────────────────┐
│  thigh/calf/foot_r quats │  thigh/calf/foot_l quats │
│  quat ×3 (12)            │  quat ×3 (12)            │
│  [0..11]                 │  [12..23]                │
└──────────────────────────┴──────────────────────────┘
```

---


## 🛠️ Key Architecture
* **NRMLPModel:** A multi-layer perceptron optimized for spatial regression.
* **NRTrainee:** Manages the backpropagation and weights persistence (`rig_model.pt`).
* **NRSolver:** Handles inference for real-time procedural feedback.

**Next Step:** Transitioning from pose-based learning to **Time-Series Prediction**, allowing the AI to generate full walking or running cycles without any external input.

![GIF](NEWRigTwo.gif)

