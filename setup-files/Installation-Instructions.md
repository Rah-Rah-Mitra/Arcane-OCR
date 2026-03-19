# Installation Instructions

Use this guide to prepare a Raspberry Pi OCR environment with local Hailo runtime artifacts.

Download the required files from:
https://hailo.ai/developer-zone/software-downloads/

Required packages:

1. TAPPAS Python Binding (`hailo_tappas_core_python_binding-5.2.0-py3-none-any.whl`)
![TAPPAS Python Binding](../docs/images/tappas_python_binding.png)

2. TAPPAS Core Ubuntu package for arm64 (`hailo-tappas-core_5.2.0_arm64.deb`)
![TAPPAS Core arm64 .deb](../docs/images/tappas_core_deb_arm64.png)

3. HailoRT PCIe driver Ubuntu package (`hailort-pcie-driver_4.23.0_all.deb`)
![HailoRT PCIe driver .deb](../docs/images/hailort_pcie_driver_deb.png)

4. HailoRT Python wheel for your Python version (aarch64)
- Python 3.10: `hailort-4.23.0-cp310-cp310-linux_aarch64.whl`
![HailoRT Python cp310 wheel](../docs/images/hailort_python_cp310_whl.png)
- Python 3.11: `hailort-4.23.0-cp311-cp311-linux_aarch64.whl` (used in this project)
![HailoRT Python cp311 wheel](../docs/images/hailort_python_cp311_whl.png)
- Python 3.12: `hailort-4.23.0-cp312-cp312-linux_aarch64.whl`
![HailoRT Python cp312 wheel](../docs/images/hailort_python_cp312_whl.png)

5. HailoRT Ubuntu package for arm64 (`hailort_4.23.0_arm64.deb`)
![HailoRT arm64 .deb](../docs/images/hailort_deb_arm64.png)

After downloading, place all required `.deb` and `.whl` files in this folder (`setup-files/`) and run:

```bash
cd /home/rahul/Development/arcane-ocr
./scripts/setup_venv.sh
source .venv/bin/activate
./scripts/install_hailo_runtime.sh
./scripts/download_models.sh
```