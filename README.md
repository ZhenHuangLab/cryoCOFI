# cryoCOFI

## Overview

cryoCOFI (CarbOn FIlm detector for cryo-EM images) is a script designed for cryo-EM images & cryo-ET tomograms to detect carbon films and get rid of particles inside them.

## Features

- Carbon film detection and particle screening in cryo-EM images
- Improved algorithm for edge detection (Bilateral filter + Canny detector, aka Bicanny)
- Integration with Dynamo data formats (.doc and .tbl files)
- GPU-accelerated image processing using CuPy and CUDA

## Requirements

- Python 3.9+
- CUDA-compatible GPU
   - CUDA Toolkit 11.1 or later
   - NVIDIA GPU Driver supporting CUDA 12.2 or later
- CuPy, >=13.3.0
- NumPy, >=2.0.2
- pandas, >=2.2.3

## Installation

### Via git clone

1. Clone the repository:
   ```
   git clone https://github.com/ZhenHuangLab/cryoCOFI.git
   ```

2. Navigate to the project directory:
   ```
   cd cryoCOFI
   ```

3. Install the package:
   ```
   pip install .
   ```

### Via pip

```
pip install cryoCOFI
```

## Usage

cryoCOFI can be used as a command-line tool:

```
cryoCOFI [command] [options]
```

Available commands:
- `readmrc`: Process a single MRC file
- `readdynamo`: Process Dynamo .doc and .tbl files

For detailed usage instructions, run:

```
cryoCOFI [command] --help
```


## License

This script is licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributing

Contributions to cryoCOFI are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please contact: zhen.victor.huang@gmail.com

For more information, visit: https://github.com/ZhenHuangLab/cryoCOFI