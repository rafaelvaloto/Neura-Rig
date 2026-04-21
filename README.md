# 🧠 NeuraRig | Neural IK & Procedural Animation

[![Status](https://img.shields.io/badge/status-active--development-green)](#)
[![C++](https://img.shields.io/badge/language-C%2B%2B20-blue)](https://en.cppreference.com/w/cpp/20)
[![LibTorch](https://img.shields.io/badge/backend-LibTorch-red)](https://pytorch.org/cppdocs/)

NeuraRig is a high-performance C++20 library dedicated to **Neural Inverse Kinematics (NIK)** and procedural character animation. It leverages deep learning to solve complex skeletal constraints in real-time, providing fluid and natural motion for characters in interactive environments.

---

## 📺 Development Progress

![NeuraRig Training Progress](NeuraRigSK_01.gif)

*The system demonstrates the learning process of the Forward Kinematics (FK) and Inverse Kinematics (IK) hybrid loop, ensuring skeletal integrity and natural joint rotations during locomotion.*

---

## 🚧 Project Status: Active Development

NeuraRig is currently in an intensive development and research stage. 

**Current Progress:**
- [x] **Core Neural Engine:** Integration with LibTorch for high-performance inference.
- [x] **Hybrid FK-IK Solver:** Specialized logic to maintain bone length and skeletal hierarchy.
- [x] **Temporal Smoothing:** Implementation of EMA and acceleration loss to eliminate jitter.
- [x] **Multi-Candidate Scoring:** Real-time selection of the best pose from multiple neural predictions.
- [ ] **Advanced Gait Logic:** Refinement of the procedural locomotion engine.
- [ ] **Extended Skeleton Support:** Scaling the system to support full-body humanoid rigs.

---

## 🛠️ Architecture & Features

### ⚙️ The Gait Engine
NeuraRig uses a programmable logic layer to define movement patterns, allowing the AI to adapt to different skeletal proportions and velocities dynamically.
- **Temporal Awareness:** Maintains continuous, jitter-free motion loops using cycle-based inputs.
- **Anatomy Agnostic:** Automatically scales strides based on bone length inputs.
- **Phase Switching:** Dynamic transitions between stance and swing phases.

### 🌟 Key Features
- **Pre-trained FK Model:** A specialized neural network that generates natural joint rotations (thigh/calf) based on end-effector positioning.
- **Physics-Informed Constraints:** Penalties for bone length variations and joint limits are integrated directly into the loss function.
- **Real-time Inference:** Optimized for low-latency performance in high-fidelity simulations.

---

## 🎮 Unreal Engine Integration

To integrate NeuraRig into Unreal Engine, use the dedicated bridge plugin:
- **Plugin Repository:** [Neura-Rig-Unreal](https://github.com/rafaelvaloto/Neura-Rig-Unreal)
- **Features:** Real-time data streaming, skeletal mapping, and seamless integration into Animation Blueprints.

---

## 📦 Dependencies & Installation

### Required Libraries
1.  **nlohmann_json:** Configuration and schema parsing.
    -   `git clone https://github.com/nlohmann/json.git 3rdParty/json`
2.  **muparser:** Gait logic expression evaluation.
    -   `git clone https://github.com/beltoforion/muparser.git 3rdParty/muparser`
3.  **LibTorch (PyTorch C++):** Neural engine.
    -   Extract into `3rdParty/libtorch`.

---

## 🏃 Getting Started

1. **Build the Project:** Use CMake to configure and build the library and tests.
2. **Model Loading:** Ensure the pre-trained model `trained_model.pt` is located in the expected directory (e.g., `Tests/Datasets/`).
3. **Run Tests:** Execute `NRTestNetwork` or `NRTestServer` to verify the installation and model performance.

---

## 📜 License
This project is licensed under the GNU General Public License v3.0. See the `LICENSE` file for details.
