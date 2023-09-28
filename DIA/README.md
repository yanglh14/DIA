# Project Name

Welcome to the DIA repository! This is where we develop and maintain our amazing application that does [describe what your application does in a few words].

![Project Logo](logo.png)

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

Provide a brief overview of your project. What is its purpose? Why was it developed? This section can also include any relevant background information or context.

## Features

List the key features of your application. This could be a bulleted list or a short paragraph describing each feature. Remember to highlight what makes your application unique or useful.

- **Feature 1:** Describe feature 1 here.
- **Feature 2:** Describe feature 2 here.
- ...
## Dataset

 - dia_baseline
 - dia_ref_final2: old version, good performance

 - dia_v2: change the time length to 2s, height to 0.3m, not work
 - dia_transfer: generalized to different shapes and stiffness
 - dia_v3: change the time length to 1s, height to 0.4m, collect_data_delta_acc~[-2,2], dia_v3_height0.4.pkl
 - dia_platform: add a platform to the bottom of the robot, dia_platform_height0.4.pkl
 - dia_platform2: release picker in last 25 steps
 - dia_platform3: not release, keep low vel in the last steps, env to large platform 
## Installation

Explain how to set up and install your project. You can include detailed steps, code snippets, or links to relevant resources. Make sure to outline any dependencies that need to be installed before your project can run successfully.

```bash
$ git clone https://github.com/your-username/project-name.git
$ cd project-name
$ npm install  # Or any other package manager commands