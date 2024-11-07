# Simple-Evasion-Adversarial-Attack
Simple Evasion Adversarial Attack

This script needs python3.10

```bash
sudo apt install python3.10 python3.10-venv python3.10-dev
python3.10 -m venv venv
source venv/bin/activate
# install dependencies
pip install -r requirements.txt
```

CLI functions to change target class or image:
```bash
# Using defaults (images/airplane.jpg and 'ship')
python adversarial_attack.py

# Specify custom image and target
python adversarial_attack.py --image path/to/image.jpg --target cat

# Get help
python adversarial_attack.py --help
```
